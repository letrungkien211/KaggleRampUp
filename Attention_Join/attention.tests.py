
from attention import ModelFactory
import unittest

class TestAttention(unittest.TestCase):
    def test_model_summary(self):
        model, _ = ModelFactory.create(
            10,
            10,
            256,
            256,
            256,
            256,
            1000,
            1000
        )
        model.summary()

if __name__ == '__main__':
    unittest.main()