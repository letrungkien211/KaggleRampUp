import tensorflow as tf
import math
import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import Dense

# Scale dot product
def scale_dot_product(Q, K, V, dk, maskStep = -1):
    x = tf.matmul(Q, K, transpose_b = True)
    x = tf.scalar_mul(1/math.sqrt(dk), x)
    if(maskStep !=-1):
        x = mask(x, maskStep)
    x = tf.nn.softmax(x)
    return tf.matmul(x, V)

def mask(input, maskStep):
    intput_shape = array_ops.shape(input)
    
    m = np.zeros(intput_shape[2], intput_shape[2])
    m[:maskStep, :maskStep] = 1
    m = tf.convert_to_tensor(m)
    return tf.matmul(input, m)   

# Multi head attention
class MultiHeadAttention():
    def __init__(self, d_model, dk, dv, Tx, h):
        self.output_dense = Dense(d_model)
        self.v_dense = Dense(dv)
        self.q_dense = Dense(dk)
        self.k_dense = Dense(dk)
        self.dk = dk
        self.dv = dv
        self.Tx = Tx
        self.h = h
        self.d_model = d_model

    def forward(self, V, K, Q):
        V = self.v_dense(V)
        Q = self.q_dense(Q)
        K = self.k_dense(K)

        output = []
        for _ in range(self.h):
            x = scale_dot_product(Q, K, V, self.dk)
            output.append(x)
        output = tf.concat(output, 2)     
        output = self.output_dense(output)
        return output

