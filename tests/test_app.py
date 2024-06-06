# tests/test_app.py
import os
import sys
import unittest
from app import app
from config.test_config import TestConfig

# Add the backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

class BasicTests(unittest.TestCase):
    def setUp(self):
        app.config.from_object(TestConfig)
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        pass

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the Image-Based Diagnostic Assistant', response.data)

    def test_upload_page(self):
        response = self.app.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload your image here', response.data)

    def test_about_page(self):
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'about the Image-Based Diagnostic Assistant', response.data)

    def test_contact_page(self):
        response = self.app.get('/contact')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'If you have any questions or concerns', response.data)

    def test_faq_page(self):
        response = self.app.get('/faq')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'answers to frequently asked questions', response.data)

    def test_privacy_page(self):
        response = self.app.get('/privacy')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'We take your privacy seriously', response.data)

    def test_terms_page(self):
        response = self.app.get('/terms')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'terms and conditions', response.data)

    def test_file_upload(self):
        with open('tests/test_image.jpg', 'rb') as test_file:
            response = self.app.post('/upload', data={'file': test_file}, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Prediction result:', response.data)

if __name__ == "__main__":
    unittest.main()
