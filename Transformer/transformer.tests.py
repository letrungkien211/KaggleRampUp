import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import unittest

from transformer import scale_dot_product, MultiHeadAttention, mask, EncoderBlock, Encoder

import tensorflow as tf
import numpy as np
import inspect

d_model = 256
dk = 5
dv = 3
Tx = 4
Ty = 7
B = 2
h = 2
d_ff = 64
maskStep = 2
Nx = 1

class TestTransformer(unittest.TestCase):
    def test_scale_product(self):
        tf.reset_default_graph();
        with tf.Session() as sess:
            Q = tf.placeholder(tf.float32, shape=(None, Ty, dk))
            K = tf.placeholder(tf.float32, shape=(None, Tx, dk))
            V = tf.placeholder(tf.float32, shape=(None, Tx, dv))
        
            output = scale_dot_product(Q, K, V, Tx, dk)
            
            q_val = np.random.rand(B, Ty, dk)
            k_val = np.random.rand(B, Tx, dk)
            v_val = np.random.rand(B, Tx, dv)

            out_val = output.eval(feed_dict={Q: q_val, K:k_val, V: v_val})
            print(inspect.stack()[0].function, '---')
            print(out_val.shape)
            print(out_val)
            assert all([x == out_val.shape[i] for i, x in enumerate([B, Ty, dv])])

    def test_mask(self):

        tf.reset_default_graph();
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, shape=(None, Ty, Ty))
            y = mask(x, Ty, Ty-2)
            out_val = y.eval(feed_dict={x: np.random.rand(B, Ty, Ty)})
            print(inspect.stack()[0].function, '---')
            print(out_val.shape)
            print(out_val)       

    def test_multi_head(self):
        tf.reset_default_graph();
        with tf.Session() as sess:
            multiHead = MultiHeadAttention(d_model, dk, dv, h)
            input = tf.placeholder(tf.float32, shape=(None, Tx, d_model))

            output = multiHead.forward(input,input,input, Tx, maskStep)

            sess.run(tf.global_variables_initializer())
            input_val = np.random.rand(B, Tx, d_model)
            out_val = output.eval(feed_dict={input: input_val})

            print(inspect.stack()[0].function, '---')
            print(out_val.shape)
            print(out_val)  
            assert all([x == out_val.shape[i] for i, x in enumerate([B, Tx, d_model])])      

    def test_encoder_model(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            Input = tf.placeholder(tf.float32, shape=(None, Ty, dk), name='Input')
            multi_head = MultiHeadAttention(d_model, dk, dv, h)
            output = multi_head.forward(Input, Input, Input, Ty, maskStep)
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./logs/encoder_model")
            writer.add_graph(sess.graph)


    # def test_dummy(self):
    #     wv = tf.get_variable('wv', (d_model, dv), dtype=tf.float32)
    #     print('OL')
    def test_draw_components(self):
        tf.reset_default_graph()

if __name__ == '__main__':
    unittest.main()