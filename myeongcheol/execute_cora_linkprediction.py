import time
import numpy as np
import tensorflow as tf

from models.gat import GAT
from utils import process

checkpt_file = 'pre_trained/cora/mod_cora_link.h5'  # 파일 확장자를 .h5로 변경

dataset = 'cora'
batch_size = 1
nb_epochs = 1000
patience = 100
lr = 0.005
l2_coef = 0.0005
hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))

# Load and preprocess the dataset
adj, features, _, _, _, _, _, _ = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]  # Update to the correct number of nodes
ft_size = features.shape[1]   # Update to the correct feature size
nb_classes = 2  # Binary classification for link prediction

adj = adj.todense()
features = features[np.newaxis]
adj = adj[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

def get_link_labels(adj):
    edges_pos = np.transpose(np.nonzero(adj))
    edges_neg = np.transpose(np.nonzero(1 - adj - np.eye(adj.shape[0])))
    labels = np.hstack([np.ones(edges_pos.shape[0]), np.zeros(edges_neg.shape[0])])
    edges = np.vstack([edges_pos, edges_neg])
    return edges, labels

edges_train, y_train = get_link_labels(adj[0])
edges_val, y_val = get_link_labels(adj[0])
edges_test, y_test = get_link_labels(adj[0])

y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]

model = GAT(nb_nodes, ft_size, nb_classes, hid_units, n_heads, residual, nonlinearity)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def train_step(features, adj, labels, edges):
    with tf.GradientTape() as tape:
        logits = model((features, adj), training=True)
        logits = tf.reshape(logits, [batch_size, nb_nodes, -1])  # Ensure the new shape matches the total number of elements
        preds = dot_product_decode(logits[0], edges)
        loss = masked_binary_cross_entropy(preds, labels[0])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = accuracy(preds, labels[0])
    return loss, acc

@tf.function
def val_step(features, adj, labels, edges):
    logits = model((features, adj), training=False)
    logits = tf.reshape(logits, [batch_size, nb_nodes, -1])  # Ensure the new shape matches the total number of elements
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
        model.save_weights(checkpt_file)  # 저장 시 확장자에 주의
        bad_counter = 0
    else:
        bad_counter += 1
    
    if bad_counter == patience:
        print('Early stopping!')
        break

# Load the best model
model.load_weights(checkpt_file)  # 로드 시 확장자에 주의

# Test the model
test_loss, test_acc = val_step(features, biases, y_test, edges_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
