from transformers import Blip2QFormerModel, MistralForCausalLM,BertTokenizer,AutoTokenizer
import torch
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"

class GraphBLIP():
    def __init__(self):
        super().__init__()
        self.config=BertConfig.from_pretrained("bert-base-uncased")
        self.word_embeddings = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, padding_idx=self.config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.num_query_token = 32
        embed_dim = 256
        self.vision_proj = nn.Linear(self.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.config.hidden_size, embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.gtm_head = nn.Linear(self.config.hidden_size, 2)
        self.qformer=Blip2QFormerModel()
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_token, self.config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,add_eos_token=True)
        self.llm_model = MistralForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        self.language_projection = nn.Linear(self.config.hidden_size, self.llm_model.config.hidden_size)

    def forward(self,graph_embeddings,caption,mode): #graph_embeddings:[batch_size,20,1024]
        if mode==1:
            text = self.tokenizer(
                caption,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            embeddings = self.word_embeddings(text.input_ids)
            query_tokens = self.query_tokens.expand(graph_embeddings.shape[0], -1, -1)
            query_output=self.qformer(query_embeds=query_tokens,encoder_hidden_states=graph_embeddings)
            graph_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
            text_output = self.qformer(query_embeds=embeddings,attention_mask=text.attention_mask)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
            ###============== Graph-text Contrastive ===================###
            sim_q2t = torch.matmul(
                graph_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size, num_query_tokens]

            # graph-text similarity: aggregate across all query tokens
            sim_g2t, _ = sim_q2t.max(-1)
            sim_g2t = sim_g2t / self.temp

            # text-query similarity: [batch_size, batch_size, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), graph_feats.permute(0, 2, 1) #permute가 맞는지?
            ).squeeze()

            # text-graph similarity: aggregate across all query tokens
            sim_t2g, _ = sim_t2q.max(-1)
            sim_t2g = sim_t2g / self.temp  # [batch_size, batch_size*num_gpu]

            bs = graph_feats.size(0)
            targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(device)

            loss_gtc = (F.cross_entropy(sim_g2t, targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2g, targets, label_smoothing=0.1)
                       ) / 2

            ###============== Graph-text Matching ===================###

            with torch.no_grad():
                sim_t2g[:, 0:bs].fill_diagonal_(-10000)
                sim_g2t[:, 0:bs].fill_diagonal_(-10000)
                weights_t2g = F.softmax(sim_t2g, dim=1)
                weights_g2t = F.softmax(sim_g2t, dim=1)

            # select a negative graph for each text
            graph_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(graph_embeddings[neg_idx])
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

            # select a negative text for each graph
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text.input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text.input_ids, text.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text.attention_mask, text.attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat(
                [graph_embeddings, graph_embeds_neg, graph_embeddings], dim=0
            )  # pos, neg, pos
            graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(
                device
            )

            output_gtm = self.qformer(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_gtm.last_hidden_state[:, : query_tokens_itm.size(1), :]
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            gtm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(device)
            loss_gtm = F.cross_entropy(logits, gtm_labels)
            loss=loss_gtm+loss_gtc
            return loss
        elif mode==2:
            query_tokens = self.query_tokens.expand(graph_embeddings.shape[0], -1, -1)
            query_output = self.qformer(query_embeds=query_tokens, encoder_hidden_states=graph_embeddings)
            input_mistral=self.language_projection(query_output.last_hidden_state)
            atts_mistral = torch.ones(input_mistral.size()[:-1], dtype=torch.long).to(device)
            self.llm_tokenizer.padding_side = "right"
            mistral_tokens = self.llm_tokenizer(
                caption,
                return_tensors="pt",
                padding="longest",
                truncation=True
            ).to(device)
            targets = mistral_tokens.input_ids.masked_fill(
                mistral_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones(atts_mistral.size(), dtype=torch.long).to(device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            inputs_embeds = self.llm_model.model.decoder.embed_tokens(mistral_tokens.input_ids)
            inputs_embeds = torch.cat([input_mistral, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_mistral, mistral_tokens.attention_mask], dim=1)
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            return loss
    def generate(self,graph_embeddings,question):
        query_tokens = self.query_tokens.expand(graph_embeddings.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeddings,
            return_dict=True,
        )
        input_mistral = self.language_projection(query_output.last_hidden_state)
        atts_mistral = torch.ones(input_mistral.size()[:-1], dtype=torch.long).to(device)
        mistral_tokens = self.llm_tokenizer(
            question,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).to(device)
        attention_mask = torch.cat([atts_mistral,mistral_tokens.attention_mask], dim=1)
        inputs_embeds = self.llm_model.get_input_embeddings()(mistral_tokens.input_ids)
        inputs_embeds = torch.cat([input_mistral, inputs_embeds], dim=1)
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=500,
            eos_token_id=2,
            early_stopping=True
        )
        output_text = self.llm_tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


class Baseline():
    def __init__(self):
        super().__init__()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_eos_token=True)
        self.llm_model = MistralForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
    def generate(self,question):
        mistral_tokens = self.llm_tokenizer(
            question,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).to(device)
        inputs_embeds = self.llm_model.get_input_embeddings()(mistral_tokens.input_ids)
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=500,
            eos_token_id=2,
            early_stopping=True
        )
        output_text = self.llm_tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
