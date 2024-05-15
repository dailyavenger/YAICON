import tensorflow as tf
import numpy as np
from models.gat import GAT
from utils import process

checkpt_file = 'pre_trained/cora/mod_cora_link.ckpt'
dataset = 'cora'

batch_size = 1
hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

# Load and preprocess the dataset
adj, features, _, _, _, _, _, _ = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = 20 #features.shape[0]
ft_size = 768 #features.shape[1]
nb_classes = 2  # Binary classification for link prediction

adj = adj.todense()
features = features[np.newaxis]
adj = adj[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

model = GAT(nb_nodes, ft_size, nb_classes, hid_units, n_heads, residual, nonlinearity)
model.load_weights(checkpt_file)

@tf.function
def get_embeddings(features, adj):
    logits = model((features, adj), training=False)
    embeddings = tf.reshape(logits, [batch_size, nb_nodes, -1])
    return embeddings

embeddings = get_embeddings(features, biases)
np.save('node_embeddings.npy', embeddings.numpy())
print("Extracted embeddings shape:", embeddings.shape)
