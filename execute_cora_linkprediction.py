import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process

checkpt_file = 'pre_trained/cora/mod_cora_link.ckpt'

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
model = GAT

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
print('model: ' + str(model))

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
    
    def dot_product_decode(embeddings, edges):
        edge_embeddings = tf.gather(embeddings, edges)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(edge_embeddings[:, 0], edge_embeddings[:, 1]), axis=1))

    preds_train = dot_product_decode(emb[0], edges_train)
    preds_val = dot_product_decode(emb[0], edges_val)
    preds_test = dot_product_decode(emb[0], edges_test)

    def masked_binary_cross_entropy(preds, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))
    
    def accuracy(preds, labels):
        correct_prediction = tf.equal(tf.round(preds), labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = masked_binary_cross_entropy(preds_train, y_train[0])
    acc = accuracy(preds_train, y_train[0])

    train_op = model.training(loss, lr, l2_coef)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        vacc_mx = 0.0
        vlss_mn = np.inf
        curr_step = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, acc],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, acc],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                     val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        
        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([loss, acc],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
