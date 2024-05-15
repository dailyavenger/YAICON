import tensorflow as tf
from models.base_gattn import BaseGAttN
from utils.layers import AttnHead

class GAT(BaseGAttN, tf.keras.Model):
    def __init__(self, nb_nodes, nb_features, nb_classes, hid_units, n_heads, residual, nonlinearity):
        super(GAT, self).__init__()
        self.attn_heads = [AttnHead(hid_units[0], nonlinearity, 0.6, 0.6, residual) for _ in range(n_heads[0])]
        self.out_layer = AttnHead(nb_classes, lambda x: x, 0.6, 0.6, residual)
        self.n_heads = n_heads
        self.nb_classes = nb_classes
        self.hid_units = hid_units
        self.residual = residual
        self.nonlinearity = nonlinearity

    def call(self, inputs, training=False):
        x, bias_mat = inputs

        attn_outputs = [attn(x, bias_mat, training=training) for attn in self.attn_heads]
        h_1 = tf.concat(attn_outputs, axis=-1)

        for i in range(1, len(self.hid_units)):
            attn_outputs = [attn(h_1, bias_mat, training=training) for attn in self.attn_heads]
            h_1 = tf.concat(attn_outputs, axis=-1)

        out = [self.out_layer(h_1, bias_mat, training=training) for _ in range(self.n_heads[-1])]
        logits = tf.add_n(out) / self.n_heads[-1]
        return logits
