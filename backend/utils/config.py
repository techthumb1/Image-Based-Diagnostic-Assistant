import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DEBUG = os.getenv('DEBUG', 'false').lower() in ['true', '1']
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS').split(','))
    PRUNED_MODEL_PATH = os.getenv('PRUNED_MODEL_PATH', 'backend/models/segformer_pruned_model')
    FEATURE_EXTRACTOR_PATH = os.getenv('FEATURE_EXTRACTOR_PATH', 'backend/models/segformer_feature_extractor')


class Config:
    def __init__(self, config_data):
        self.config = config_data

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Config(config_data)

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def apply_config(config):
    for key, value in config.config.items():
        setattr(Config, key, value)

        