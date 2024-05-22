import json
import numpy as np
import torch
import torch.nn as nn


class GATtoQformer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATtoQformer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


batch_size = 64
i = 0

# JSON 파일 경로
node_embeddings_path = 'node_embeddings_f.json'
kg2text_path = 'kg2text.json'
subgraph_path = 'subgraph_list_string.json'

with open(node_embeddings_path, 'r') as f:
    node_embedding = json.load(f)
with open(kg2text_path, 'r') as f:
    kg2text = json.load(f)
with open(subgraph_path, 'r') as f:
    subgraph = json.load(f)

graph_embeddings = []
gat_to_qformer = GATtoQformer(in_features=1024, out_features=768)

for x in node_embedding:
    graph_id = x['graph_id']
    embeddings = x['embeddings']
    subgraph_text = subgraph[i]

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    gat_embeddings = embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
    gat_embeddings = gat_to_qformer(gat_embeddings)
    print(gat_embeddings.shape)

    new_item = {
        'graph_id': graph_id,
        'embeddings': gat_embeddings.tolist(),  # Convert back to list if needed for JSON serialization
        'subgraph_text': subgraph_text
    }

    graph_embeddings.append(new_item)
    i += 1

with open('graph_embeddings.json', 'w') as f:
    json.dump(graph_embeddings, f, indent=4)
