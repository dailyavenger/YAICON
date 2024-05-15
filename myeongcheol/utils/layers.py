import tensorflow as tf

class AttnHead(tf.keras.layers.Layer):
    def __init__(self, out_sz, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        super(AttnHead, self).__init__()
        self.out_sz = out_sz
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.conv1 = tf.keras.layers.Conv1D(out_sz, 1, use_bias=False)
        self.conv2 = tf.keras.layers.Conv1D(1, 1)
        self.conv3 = tf.keras.layers.Conv1D(1, 1)
        self.bias_add = tf.keras.layers.Add()

    def call(self, seq, bias_mat, training=False):
        if self.in_drop != 0.0:
            seq = tf.nn.dropout(seq, rate=self.in_drop if training else 0.0)

        seq_fts = self.conv1(seq)

        f_1 = self.conv2(seq_fts)
        f_2 = self.conv3(seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if self.coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, rate=self.coef_drop if training else 0.0)
        if self.in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, rate=self.in_drop if training else 0.0)

        vals = tf.matmul(coefs, seq_fts)
        ret = self.bias_add([vals])

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.keras.layers.Conv1D(ret.shape[-1], 1)(seq)
            else:
                ret = ret + seq

        return self.activation(ret)
