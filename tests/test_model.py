import unittest
from models.model import create_model

class ModelTestCase(unittest.TestCase):
    def test_model_load(self):
        """Test model loading."""
        self.assertIsInstance(create_model(), tf.keras.models.Model)

    def test_prediction(self):
        """Test model prediction."""
        model = create_model()
        test_input = tf.random.normal([1, 224, 224, 3])
        prediction = model.predict(test_input)
        self.assertEqual(prediction.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
