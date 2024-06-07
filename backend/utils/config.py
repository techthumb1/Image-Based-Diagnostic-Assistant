import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS').split(','))
    MODEL_PATH = os.getenv('MODEL_PATH')

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)