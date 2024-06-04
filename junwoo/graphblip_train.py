import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm, trange
import json
from graphblip import GraphBLIP, Baseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as file:
            self.data = json.load(file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        graph_embedding = torch.tensor(item['embeddings'])
        caption = item['subgraph_text']
        return graph_embedding, caption

def train_model(model, dataloader, optimizer, scheduler, epochs):
    model.train()
    for epoch in trange(epochs, desc="Epochs"):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Batches", leave=False)
        for batch in progress_bar:
            graph_embeddings, captions = batch
            graph_embeddings = graph_embeddings.to(device)
            captions = list(captions)
            optimizer.zero_grad()
            loss = model(graph_embeddings, captions, mode=1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (progress_bar.n + 1)})
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")
        torch.save(model.state_dict(), f'graph_blip_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    dataset = CustomDataset('graph_embeddings_3.json')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = GraphBLIP()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(dataloader) * 5
#     num_training_steps = min(len(dataloader) * 5, 10)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train_model(model, dataloader, optimizer, scheduler, epochs=5)
