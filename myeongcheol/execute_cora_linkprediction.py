import numpy as np
import tensorflow as tf
import json
import tensorflow_gnn as tfgnn
import os

from models.gat import GAT
from utils import process

# Load files
checkpt_file = 'pre_trained/gat_Encoder.h5'
entity2id_file = 'graph_embed/entity2id.txt'
entity_embed_file = 'graph_embed/entity_embed.json'
subgraph_list_adj2_file = 'graph_embed/subgraph_list_adj2.json'
subgraph_list_entity2_file = 'graph_embed/subgraph_list_entity2.json'

# Load entity2id
entity2id = {}
with open(entity2id_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            entity, eid = parts
            entity2id[int(eid)] = entity

# Load entity embeddings
with open(entity_embed_file, 'r') as f:
    entity_embed = json.load(f)

# Load subgraph adjacency matrices
with open(subgraph_list_adj2_file, 'r') as f:
    subgraph_list_adj2 = json.load(f)

# Load subgraph entity lists
with open(subgraph_list_entity2_file, 'r') as f:
    subgraph_list_entity2 = json.load(f)

# Convert subgraphs to GraphTensors
subgraphs = []
for triples, entity_ids in zip(subgraph_list_adj2, subgraph_list_entity2):
    num_nodes = len(entity_ids)
    node_features = []

    for entity_id in entity_ids:
        embedding_vector = entity_embed.get(str(entity_id))
        if embedding_vector is not None:
            node_features.append(embedding_vector)
        else:
            node_features.append(np.zeros(32))

    node_features = np.array(node_features).astype(np.float32)
    edge_indices = []
    edge_types = []

    for triple in triples:
        head, rel, tail = triple
        head_idx = entity_ids.index(head)
        tail_idx = entity_ids.index(tail)
        edge_indices.append([head_idx, tail_idx])
        edge_types.append(rel)

    edge_indices = np.array(edge_indices).T
    edge_indices = [tf.convert_to_tensor(edge_indices[0], dtype=tf.int32), tf.convert_to_tensor(edge_indices[1], dtype=tf.int32)]
    edge_types = tf.convert_to_tensor(edge_types, dtype=tf.int32)
    node_features = tf.convert_to_tensor(node_features, dtype=tf.float32)
    
    # Normalize node features
    node_features = tf.keras.utils.normalize(node_features, axis=-1)

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'nodes': tfgnn.NodeSet.from_fields(
                sizes=[num_nodes],
                features={'feat': node_features}
            )
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                sizes=[len(edge_indices[0])],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('nodes', edge_indices[0]),
                    target=('nodes', edge_indices[1])
                ),
                features={'type': edge_types}
            )
        }
    )
    subgraphs.append(graph)

# GAT model configuration
batch_size = 10
nb_epochs = 10000
patience = 100
lr = 0.01
l2_coef = 0.0005
hid_units = [64]
n_heads = [64, 1]
residual = False
nonlinearity = tf.nn.elu

print('----- 최적화 하이퍼파라미터 -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- 아키텍처 하이퍼파라미터 -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))

# Prepare dataset from subgraphs
features = []
adj_list = []
for graph in subgraphs:
    node_features = graph.node_sets['nodes']['feat']
    edge_sources = graph.edge_sets['edges'].adjacency.source
    edge_targets = graph.edge_sets['edges'].adjacency.target
    adj_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))
    
    edge_sources_np = edge_sources.numpy()
    edge_targets_np = edge_targets.numpy()
    
    for i, j in zip(edge_sources_np, edge_targets_np):
        adj_matrix[i, j] = 1
    
    features.append(node_features.numpy())
    adj_list.append(adj_matrix)

features = np.array(features)
adj_list = np.array(adj_list)

print("feature dimension:", features.shape)
print("adj_list dimension:", adj_list.shape)

# Process adjacency matrices
biases = process.adj_to_bias(adj_list, [adj.shape[0] for adj in adj_list], nhood=1)

def get_link_labels(adj):
    edges_pos = np.transpose(np.nonzero(adj))
    edges_neg = np.transpose(np.nonzero(1 - adj - np.eye(adj.shape[0])))
    labels = np.hstack([np.ones(edges_pos.shape[0]), np.zeros(edges_neg.shape[0])])
    edges = np.vstack([edges_pos, edges_neg])
    return edges, labels

edges_train, y_train = get_link_labels(adj_list[0])
edges_val, y_val = get_link_labels(adj_list[0])
edges_test, y_test = get_link_labels(adj_list[0])

y_train = y_train[np.newaxis].astype(np.float32)
y_val = y_val[np.newaxis].astype(np.float32)
y_test = y_test[np.newaxis].astype(np.float32)

model = GAT(features.shape[1], features.shape[2], 2, hid_units, n_heads, residual, nonlinearity)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory='checkpoints', max_to_keep=3)

# Define the callback for saving checkpoints
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)

@tf.function
def train_step(features, adj, labels, edges):
    with tf.GradientTape() as tape:
        logits = model((features, adj), training=True)
        logits = tf.reshape(logits, [batch_size, features.shape[1], -1])
        preds = dot_product_decode(logits[0], edges)
        loss = masked_binary_cross_entropy(preds, labels[0])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = accuracy(preds, labels[0])
    return loss, acc

@tf.function
def val_step(features, adj, labels, edges):
    logits = model((features, adj), training=False)
    logits = tf.reshape(logits, [batch_size, features.shape[1], -1])
    preds = dot_product_decode(logits[0], edges)
    loss = masked_binary_cross_entropy(preds, labels[0])
    acc = accuracy(preds, labels[0])
    return loss, acc

@tf.function
def dot_product_decode(embeddings, edges):
    edge_embeddings = tf.gather(embeddings, edges)
    return tf.sigmoid(tf.reduce_sum(tf.multiply(edge_embeddings[:, 0], edge_embeddings[:, 1]), axis=1))

def masked_binary_cross_entropy(preds, labels):
    labels = tf.cast(labels, tf.float32)  
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.round(preds), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training loop with checkpoint saving
best_val_loss = float('inf')
bad_counter = 0

for epoch in range(nb_epochs):
    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0
    
    train_loss, train_acc = train_step(features, biases, y_train, edges_train)
    train_loss_avg += train_loss
    train_acc_avg += train_acc
    
    val_loss, val_acc = val_step(features, biases, y_val, edges_val)
    val_loss_avg += val_loss
    val_acc_avg += val_acc
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}')
    
    # Save the model checkpoint at the end of every epoch
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        model.save_weights(checkpt_file)
        checkpoint_manager.save()
        bad_counter = 0
    else:
        bad_counter += 1
    
    if bad_counter == patience:
        print('Early stopping!')
        break

model.load_weights(checkpt_file)
checkpoint.restore(checkpoint_manager.latest_checkpoint)

test_loss, test_acc = val_step(features, biases, y_test, edges_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Function to get node embeddings
def get_node_embeddings(model, features, adj):
    embeddings = model.get_node_embeddings((features, adj), training=False)
    return embeddings

# Extract node embeddings for each graph
all_node_embeddings = []
for i in range(len(features)):
    node_embeddings = get_node_embeddings(model, features[i:i+1], biases[i:i+1])
    all_node_embeddings.append({"graph_id": i, "embeddings": node_embeddings.numpy().tolist()})

# Save node embeddings to JSON
with open('node_embeddings.json', 'w') as f:
    json.dump(all_node_embeddings, f, indent=4)  # Using indent=4 for readability

print("Node embeddings saved to node_embeddings.json")
