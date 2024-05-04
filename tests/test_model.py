import unittest
from model.cnn_model import model_predict

class TestModelPredict(unittest.TestCase):
    def test_prediction_output(self):
        # Assume you have a sample image path and a function to handle prediction
        result = model_predict('path_to_sample_image.jpg')
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
