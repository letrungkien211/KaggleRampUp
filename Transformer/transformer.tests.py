import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import unittest

from transformer import scale_dot_product, MultiHeadAttention, mask

import tensorflow as tf
import numpy as np
import inspect

sess = tf.InteractiveSession()

d_model = 10
dk = 5
dv = 3
Tx = 4
Ty = 7
B = 2
h = 6
maskStep = 2

class TestTransformer(unittest.TestCase):
    def test_scale_product(self):
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
        x = tf.placeholder(tf.float32, shape=(None, dk, dk))
        y = mask(x, dk, dk-1)
        out_val = y.eval(feed_dict={x: np.random.rand(B, dk, dk)})
        print(inspect.stack()[0].function, '---')
        print(out_val.shape)
        print(out_val)       

    def test_multi_head(self):
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

    # def test_dummy(self):
    #     wv = tf.get_variable('wv', (d_model, dv), dtype=tf.float32)
    #     print('OL')

if __name__ == '__main__':
    unittest.main()