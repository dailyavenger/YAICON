import numpy as np
import tensorflow as tf
import json
import tensorflow_gnn as tfgnn

from models.gat import GAT
from utils import process

# File paths
checkpt_file = 'pre_trained/gat_encoder.h5'  # Save file path for the model
entity2id_file = 'graph_embed/entity2id.txt'
entity_embed_file = 'graph_embed/entity_embed.json'
subgraph_list_adj2_file = 'graph_embed/subgraph_list_adj2.json'
subgraph_list_entity2_file = 'graph_embed/subgraph_list_entity2.json'

# Load entity2id file
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
print(subgraph_list_adj2)

# Load subgraph entity lists
with open(subgraph_list_entity2_file, 'r') as f:
    subgraph_list_entity2 = json.load(f)
print(subgraph_list_entity2)

# Convert subgraph adjacency matrices and entity lists to GraphTensor
subgraphs = []
for adj_matrix, entity_ids in zip(subgraph_list_adj2, subgraph_list_entity2):
    adj_matrix = np.array(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    edge_indices = np.nonzero(adj_matrix)
    
    node_features = []
    for entity_id in entity_ids:
        mid = entity2id.get(entity_id)
        embedding_vector = entity_embed.get(mid)
        if embedding_vector is not None:
            node_features.append(embedding_vector)
        else:
            node_features.append(np.zeros(600))  # Use zero vector if embedding is not found

    node_features = np.array(node_features).astype(np.float32)
    edge_indices = [tf.convert_to_tensor(edge_indices[0], dtype=tf.int32), tf.convert_to_tensor(edge_indices[1], dtype=tf.int32)]
    node_features = tf.convert_to_tensor(node_features, dtype=tf.float32)
    
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
                )
            )
        }
    )
    subgraphs.append(graph)

# GAT model setup and training
batch_size = 1
nb_epochs = 1000
patience = 100
lr = 0.005
l2_coef = 0.0005
hid_units = [8]  # Numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # Additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
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
    edge_indices = graph.edge_sets['edges'].adjacency.source
    adj_matrix = np.zeros((node_features.shape[0], node_features.shape[0]))
    edge_indices_np = edge_indices.numpy()
    for i, j in zip(edge_indices_np[0], edge_indices_np[1]):
        adj_matrix[i, j] = 1
    features.append(node_features.numpy())
    adj_list.append(adj_matrix)
    
features = np.array(features)
adj_list = np.array(adj_list)

print("Features shape:", features.shape)
print("Adjacency list shape:", adj_list.shape)

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

y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]

model = GAT(features.shape[1], features.shape[2], 2, hid_units, n_heads, residual, nonlinearity)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.round(preds), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training loop
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
    
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        model.save_weights(checkpt_file)  # Save the model weights
        bad_counter = 0
    else:
        bad_counter += 1
    
    if bad_counter == patience:
        print('Early stopping!')
        break

# Load the best model
model.load_weights(checkpt_file)  # Load the saved model weights

# Test the model
test_loss, test_acc = val_step(features, biases, y_test, edges_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')