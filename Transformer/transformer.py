import tensorflow as tf
import math
import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import Dense

# Scale dot product
def scale_dot_product(Q, K, V, dk, T = -1, maskStep = -1):
    x = tf.matmul(Q, K, transpose_b = True)
    x = tf.scalar_mul(1/math.sqrt(dk), x)
    if(maskStep !=-1 and T!=-1):
        x = mask(x, T, maskStep)
    x = tf.nn.softmax(x)
    return tf.matmul(x, V)

# mask
def mask(input, dim,  i):
    m = tf.ones([i, i])
    m = tf.pad(m, [[0, dim-i], [0, dim-i]])
    return tf.multiply(input, m)

class H_Layer():
    def __init__(self, dk, dv):
        with tf.variable_scope('H_Layer'):
            self.v_dense = Dense(dv, name='Wv')
            self.q_dense = Dense(dk, name='Wq')
            self.k_dense = Dense(dk, name='Wk')
        self.dk = dk
    def forward(self, V, K, Q, T = -1, maskStep = -1):
        V = self.v_dense(V)
        Q = self.q_dense(Q)
        K = self.k_dense(K)
        return scale_dot_product(Q, K, V, self.dk, T, maskStep)

# Multi head attention
class MultiHeadAttention():
    def __init__(self, d_model, dk, dv, h):
        self.output_dense = Dense(d_model)
        self.h_layers = [H_Layer(dk, dv) for i in range(h)]

    def forward(self, V, K, Q, T=-1, maskStep=-1):
        output = [layer.forward(V, K, Q, T, maskStep) for layer in self.h_layers]
        output = tf.concat(output, 2)     
        output = self.output_dense(output)
        return output

class AddNorm():
    def __init__(self):
        pass
    def forward(self, x, y):
        ## TODO: Add layer normalization layer. tf.keras.layers doesn't have yet. May need to use the one in tf.contrib
        return tf.add(x, y)

class FFN():
    def __init__(self, d_model, d_ff):
        self.dense_1 = Dense(d_ff, activation='relu')
        self.dense_2 = Dense(d_model)
    def forward(self, x):
        return self.dense_2(self.dense_1(x))

# Encoder block
class EncoderBlock():
    def __init__(self, d_model, d_ff, dk, dv, h):
        self.mha = MultiHeadAttention(d_model, dk, dv, h)
        self.addnorm_1 = AddNorm()
        self.ffn = FFN(d_model, d_ff)
        self.addnorm_2 = AddNorm()
    def forward(self, input):
        x = self.mha.forward(input, input, input)
        x = self.addnorm_1.forward(x, input)
        
        y = self.ffn.forward(x)
        y = self.addnorm_2.forward(x, y)

        return y

class Encoder():
    def __init__(self, Nx, encoder_block):
        self.encoder_block = encoder_block
        self.Nx = Nx
    def forward(self, input):
        output = input
        for _ in range(self.Nx):
            output = self.encoder_block.forward(output)
        return output

class DecoderBlock():
    def __init__(self, d_model, d_ff, dk, dv, h):
        self.mha_1 = MultiHeadAttention(d_model, dk, dv, h)
        self.addnorm_1 = AddNorm()
        self.mha_2 = MultiHeadAttention(d_model, dk, dv, h)
        self.addnorm_2 = AddNorm()
        self.ffn = FFN(d_model, d_ff)
        self.addnorm_3 = AddNorm()

    def forward(self, decoder_input, encoder_output, Ty, maskStep):
        x = decoder_input
        y = self.mha_1.forward(x, x, x, Ty, maskStep)
        x = self.addnorm_1.forward(x, y)

        y = self.mha_2.forward(encoder_output, encoder_output, x)
        x = self.addnorm_2.forward(x, y)

        y = self.ffn.forward(x)
        x = self.addnorm_3.forward(x, y)

class Decoder():
    def __init__(self,Nx, decoder_block):
        self.decoder_block = decoder_block
        self.Nx = Nx
    def forward(self, decoder_input, encoder_output, Ty, maskStep):
        output = decoder_input
        for _ in range(self.Nx):
            output = self.decoder_block.forward(output, encoder_output, Ty, maskStep)
        return output