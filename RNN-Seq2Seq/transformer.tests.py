import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import unittest

from transformer import ScaleDotProduct, MultiHeadAttention

import tensorflow as tf
import numpy as np
import inspect

sess = tf.InteractiveSession()

d_model = 10
dk = 5
dv = 3
Tx = 4
B = 2
h = 3

class TestTransformer(unittest.TestCase):
    def test_scale_product(self):
        scaledot = ScaleDotProduct()

        Q = tf.placeholder(tf.float32, shape=(None, Tx, dk))
        K = tf.placeholder(tf.float32, shape=(None, Tx, dk))
        V = tf.placeholder(tf.float32, shape=(None, Tx, dv))
       
        output = scaledot.forward(Q, K, V, dk, Tx, True, 2)
        
        q_val = np.random.rand(B, Tx, dk)
        k_val = np.random.rand(B, Tx, dk)
        v_val = np.random.rand(B, Tx, dv)

        out_val = output.eval(feed_dict={Q: q_val, K:k_val, V: v_val})
        print(inspect.stack()[0].function, '---')
        print(out_val.shape)
        print(out_val)
        assert all([x == out_val.shape[i] for i, x in enumerate([B, Tx, dv])])      

    def test_multi_head(self):
        multiHead = MultiHeadAttention(d_model, dk, dv, Tx, True, Tx-1, h)
        input = tf.placeholder(tf.float32, shape=(None, Tx, d_model))

        output = multiHead.forward(input,input,input)

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