
from attention import ModelFactory
import unittest
from keras.utils import plot_model

class TestAttention(unittest.TestCase):
    def test_model_summary(self):
        model, _ = ModelFactory.create(
            10,
            1,
            256,
            256,
            256,
            256,
            1000,
            1000
        )
        print(model.count_params())
        # model.summary()
        plot_model(model, to_file='../data/tests/model_3.pdf')
        print(model.get_layer('EncoderBiRnn').output.shape)
        
        model, _ = ModelFactory.create(
            10,
            2,
            256,
            256,
            256,
            256,
            1000,
            1000
        )
        # model.summary()
        plot_model(model, to_file='../data/tests/model_2.pdf')
        print(model.count_params())
        print(model.get_layer('EncoderBiRnn').output.shape)



if __name__ == '__main__':
    unittest.main()