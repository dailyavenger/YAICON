"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import tensorflow as tf
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from registry import registry
from base_model import all_gather_with_grad, concat_all_gather
from blip.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from blip.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        # >>> from models import load_model
        # >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "train/blip2_pretrain.yaml",
    }
    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            ft_size=768,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, ft_size, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, samples):
        gat = samples["embeddings"]
        text = samples["subgraph_text"]

        gat_embeddings = torch.tensor(gat, dtype=torch.float32).to(self.device)
        gat_atts = torch.ones(gat_embeddings.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                       :gat_embeddings.size(1), :
                       ].expand(gat_embeddings.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=gat_embeddings,
            encoder_attention_mask=gat_atts,
            use_cache=True,
            return_dict=True,
        )

        graph_feats = F.normalize(
            self.graph_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Graph-text Contrastive ===================###
        graph_feats_all = concat_all_gather(
            graph_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            graph_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # graph-text similarity: aggregate across all query tokens
        sim_g2t, _ = sim_q2t.max(-1)
        sim_g2t = sim_g2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), graph_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-graph similarity: aggregate across all query tokens
        sim_t2g, _ = sim_t2q.max(-1)
        sim_t2g = sim_t2g / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = gat_embeddings.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        #     image.device
        # )
        targets = torch.arange(rank * bs, rank * bs + bs, dtype=int).to(self.device)

        loss_itc = (
                           F.cross_entropy(sim_g2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2g, targets, label_smoothing=0.1)
                   ) / 2

        ###============== Graph-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        gat_embeds_world = all_gather_with_grad(gat_embeddings)
        with torch.no_grad():
            sim_t2g[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
            sim_g2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2g = F.softmax(sim_t2g, dim=1)
            weights_g2t = F.softmax(sim_g2t, dim=1)

        # select a negative image for each text
        graph_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2g[b], 1).item()
            graph_embeds_neg.append(gat_embeds_world[neg_idx])
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_g2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_gtm = self.Qformer.embeddings.word_embeddings.weight[:gat_embeddings.size(1), :].expand(gat_embeddings.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_gtm.size()[:-1], dtype=torch.long).to(self.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        graph_embeds_all = torch.cat(
            [gat_embeddings, graph_embeds_neg, gat_embeddings], dim=0
        )  # pos, neg, pos
        graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(
            self.device
        )

        output_gtm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_gtm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=graph_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_gtm.last_hidden_state[:, : query_tokens_gtm.size(1), :]
        vl_output = self.gtm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        gtm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_gtm = F.cross_entropy(logits, gtm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_gtm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=30,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        gat = samples["embeddings"]
        gat_embeddings = torch.tensor(gat, dtype=torch.float32).to(self.device)

        if not use_nucleus_sampling:
            gat_embeddings = gat_embeddings.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1

        gat_atts = torch.ones(gat_embeddings.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": gat_embeddings,
            "encoder_attention_mask": gat_atts,
        }

        input_ids = (
            torch.LongTensor(gat_embeddings.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(gat_embeddings.device)
        )
        query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                       : gat_embeddings.size(1), :
                       ].expand(gat_embeddings.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_graph(self, gat):
        gat_embeddings = torch.tensor(gat, dtype=torch.float32).to(self.device)
        gat_atts = torch.ones(gat_embeddings.size()[:-1], dtype=torch.long).to(
            self.device
        )

        query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                       : gat_embeddings.size(1), :
                       ].expand(gat_embeddings.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=gat_embeddings,
            encoder_attention_mask=gat_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, gat_embeddings

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_gtm(self, gat_embeddings, text_ids, text_atts):
        gat_atts = torch.ones(gat_embeddings.size()[:-1], dtype=torch.long).to(
            self.device
        )
        query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                       : gat_embeddings.size(1), :
                       ].expand(gat_embeddings.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=gat_embeddings,
            encoder_attention_mask=gat_atts,
            return_dict=True,
        )
        vl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :]
        gtm_logit = self.gtm_head(vl_embeddings)
        gtm_logit = gtm_logit[:, :, 1].mean(dim=1)
        return gtm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        gat = samples.get("graph")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "graph",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        gat_embedding, text_embeds, multimodal_embeds = None, None, None
        graph_features, text_features = None, None

        if mode == "graph":
            assert (
                    gat is not None
            ), "Graph is not provided for mode 'graph' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                graph_embeds_frozen = torch.tensor(gat, dtype=torch.float32).to(self.device)
            graph_embeds_frozen = graph_embeds_frozen.float()
            gat_atts = torch.ones(graph_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                           : graph_embeds_frozen.size(1), :
                           ].expand(graph_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds_frozen,
                encoder_attention_mask=gat_atts,
                return_dict=True,
            )
            graph_embeds = query_output.last_hidden_state
            graph_features = F.normalize(self.graph_proj(graph_embeds), dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                graph_embeds_frozen = torch.tensor(gat, dtype=torch.float32).to(self.device)
            graph_embeds_frozen = graph_embeds_frozen.float()
            gat_atts = torch.ones(graph_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.Qformer.embeddings.word_embeddings.weight[
                           : graph_embeds_frozen.size(1), :
                           ].expand(graph_embeds_frozen.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=graph_embeds_frozen,
                encoder_attention_mask=gat_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            graph_embeds=graph_embeds,
            graph_embeds_proj=graph_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            # vit_model=vit_model,
            # img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            # freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity g2t, t2g matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

