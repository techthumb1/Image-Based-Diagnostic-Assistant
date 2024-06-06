def test_invalid_file_upload(self):
    with open('tests/test_invalid_file.txt', 'rb') as test_file:
        response = self.app.post('/upload', data={'file': test_file}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'No selected file', response.data)
