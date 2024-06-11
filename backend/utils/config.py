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

def apply_config(app):
    app.config.from_object(Config)
    
def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)