def test_prediction_failure(self):
    with unittest.mock.patch('app.model_predict', side_effect=Exception("Model prediction failed")):
        with open('tests/test_image.jpg', 'rb') as test_file:
            response = self.app.post('/upload', data={'file': test_file}, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'An error occurred while processing the file.', response.data)
