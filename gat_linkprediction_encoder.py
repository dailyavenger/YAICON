import numpy as np
import tensorflow as tf
from models import GAT
from utils import process

# Checkpoint file from the previously trained GAT model
checkpt_file = 'pre_trained/cora/mod_cora_link.ckpt'  # 학습 결과로 바꾸기.

# Dataset and model hyperparameters
dataset = 'cora'
batch_size = 1
hid_units = [8]  # Number of hidden units per each attention head in each layer
n_heads = [8, 1]  # Additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)

# Load and preprocess the dataset
adj, features, _, _, _, _, _, _ = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = 20 #features.shape[0]
ft_size = 768 #features.shape[1]

adj = adj.todense()
features = features[np.newaxis]
adj = adj[np.newinstance]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

# Define the TensorFlow graph
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_nodes, nb_nodes, is_train,
                             attn_drop, ffd_drop,
                             bias_mat=bias_in,
                             hid_units=hid_units, n_heads=n_heads,
                             residual=residual, activation=nonlinearity)

    emb = tf.reshape(logits, [batch_size, nb_nodes, -1])

    # Saver to load the trained model
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Start a TensorFlow session
    with tf.Session() as sess:
        sess.run(init_op)
        
        # Restore the trained GAT model
        saver.restore(sess, checkpt_file)
        
        # Extract embeddings
        embeddings = sess.run(emb, feed_dict={
            ftr_in: features,
            bias_in: biases,
            is_train: False,
            attn_drop: 0.0, ffd_drop: 0.0})

        print("Extracted embeddings shape:", embeddings.shape)
        
        # Save the embeddings to a file
        np.save('node_embeddings.npy', embeddings[0])
        
        sess.close()
