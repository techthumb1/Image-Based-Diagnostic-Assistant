# config/test_config.py
class TestConfig:
    TESTING = True
    DEBUG = True
    SECRET_KEY = 'test_secret_key'
    UPLOAD_FOLDER = 'test_uploads'
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    MODEL_PATH = 'models/path_to_model/model.h5'
