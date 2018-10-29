import tensorflow as tf
import math
import numpy as np

# Symbols
def scale_dot_product(Q, K, V, dk):
    x = tf.matmul(Q, K, transpose_b = True)
    x = tf.scalar_mul(1/math.sqrt(dk), x)
    x = tf.nn.softmax(x)
    return tf.matmul(x, V)

# Multi head attention
class MultiHeadAttention():
    def __init__(self, d_model, dk, dv, Tx, h):
        self.dk = dk
        self.dv = dv
        self.Tx = Tx
        self.h = h
        self.d_model = d_model

    def forward(self, V, K, Q):
        V = tf.layers.dense(V, self.dv)
        Q = tf.layers.dense(Q, self.dk)
        K = tf.layers.dense(K, self.dk)

        output = []
        for _ in range(self.h):
            x = scale_dot_product(Q, K, V, self.dk)
            output.append(x)
        output = tf.concat(output, 2)
        
        output = tf.layers.dense(output, self.d_model)

        return output