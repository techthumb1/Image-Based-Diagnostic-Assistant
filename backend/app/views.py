import os
import numpy as np
import logging
import torch
import yaml
from flask import Blueprint
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
from safetensors.torch import load_file
from utils.config import Config, load_yaml_config, apply_config
from app.load_preproces import allowed_file
from backend.models.model_training import get_classification_model, get_unetpp_model, get_efficientnet_model
from preprocessing.data_loader import preprocess_image, transform
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from huggingface_hub import login
from safetensors.torch import load_file
from backend.models.unetpp import UNetPP, load_and_preprocess_image, model_predict
from efficientnet_pytorch import EfficientNet
from metrics.metrics import (
    calculate_metrics,
    dice_coefficient,
    intersection_over_union,
    pixel_accuracy,
    mean_accuracy,
    mean_iou,
    boundary_f1_score
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

app = Flask(__name__)
blueprint = Blueprint('app', __name__)


# Load configuration
base_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_path, '../config/config.yaml')

try:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Config file not found at {config_path}")


config = config_data
app.config.update(config)

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['ALLOWED_EXTENSIONS'] = set(os.getenv('ALLOWED_EXTENSIONS').split(','))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY')
app.debug = os.getenv('DEBUG', 'false').lower() in ['true', '1']
app.env = 'development'

apply_config(app)

db = SQLAlchemy(app)


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for SQLAlchemy
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Set the Hugging Face token
hf_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=hf_token, add_to_git_credential=True)
logger.info("Login successful")

# Load configuration
config_path = 'backend/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Set global variables from the configuration
model_type = config.get('MODEL_TYPE', 'classification')
model_dir = config.get('MODEL_DIR', 'models/')
pruned_model_path = config['pruned_model_path']
feature_extractor_path = config['feature_extractor_path']
labels_folder = config['labels']['folder']

# Initialize models and feature extractor
classification_model, segmentation_model, unetpp_model, feature_extractor = None, None, None, None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    if model_type == 'classification':
        logger.info("Loading the EfficientNet model for classification...")
        classification_model = EfficientNet.from_name('efficientnet-b0')  # Adjust the architecture name as needed
        
        # Load the state dictionary
        state_dict = torch.load(os.path.join(model_dir, 'effnet_b3_model_best.pth'), map_location=torch.device('cpu'))
        model_path = os.path.join(model_dir, 'effnet_b3_model_best.pth')

        
        # If the state_dict keys are prefixed with 'model.', you need to remove the prefix
        if list(state_dict.keys())[0].startswith('model.'):
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
        
        classification_model.load_state_dict(state_dict, strict=False)
        classification_model.eval()
        classification_model.to(device)
        logger.info("EfficientNet model loaded successfully")

    elif model_type == 'segmentation':
        logger.info("Loading the Segformer model for segmentation...")
        config = SegformerConfig.from_pretrained(feature_extractor_path)
        state_dict = load_file(pruned_model_path)
        segmentation_model = SegformerForSemanticSegmentation(config)
        segmentation_model.load_state_dict(state_dict)
        segmentation_model.eval()
        segmentation_model.to(device)
        logger.info("Segmentation model loaded successfully")
    elif model_type == 'unetpp':
        logger.info("Loading the UNet++ model...")
        unetpp_model = get_unetpp_model()
        unetpp_model.load_state_dict(torch.load(os.path.join(model_dir, 'unetpp_model.pth'), map_location=torch.device('cpu')))
        unetpp_model.eval()
        unetpp_model.to(device)
        logger.info("UNet++ model loaded successfully")
    else:
        raise ValueError("Unsupported model type specified in configuration.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    logger.info("Loading the feature extractor...")
    feature_extractor = SegformerImageProcessor.from_pretrained(feature_extractor_path)
    logger.info("Feature extractor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load feature extractor: {e}")
    raise

def preprocess_image(image_path):
    from PIL import Image
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# Prediction function
def classification_predict(image_tensor):
    logger.info("Starting classification prediction...")
    inputs = feature_extractor(images=image_tensor, return_tensors="pt", do_rescale=False)['pixel_values'].to(device)
    logger.debug(f"Inputs to model: {inputs}")
    
    with torch.no_grad():
        outputs = classification_model(inputs)
    
    # Handle outputs directly
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    logger.debug(f"Model logits: {logits}")
    
    predicted_class_idx = logits.argmax(-1).item()
    confidence_score = np.max(torch.nn.functional.softmax(logits, dim=-1).cpu().numpy())
    logger.info(f'Predicted class index: {predicted_class_idx}, Confidence score: {confidence_score}')
    return predicted_class_idx, confidence_score


def segmentation_predict(image_tensor, model_type):
    logger.info("Starting segmentation prediction...")
    inputs = feature_extractor(images=image_tensor, return_tensors="pt", do_rescale=False)['pixel_values'].to(device)
    
    if model_type == 'unetpp':
        model = unetpp_model
    else:
        model = segmentation_model

    with torch.no_grad():
        outputs = model(inputs)

    predicted_mask = outputs.argmax(dim=1).squeeze().cpu().numpy()
    logger.info(f'Predicted mask shape: {predicted_mask.shape}')
    return predicted_mask

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Username may already be taken.', 'danger')
            logger.error(f"Registration error: {e}")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

def load_ground_truth_label(image_path, model_type):
    base_name = os.path.basename(image_path).split('.')[0]
    if model_type == 'classification':
        label_path = os.path.join('labels', f'{base_name}_label.txt')
    elif model_type in ['segmentation', 'unetpp']:
        label_path = os.path.join('labels', f'{base_name}_label.png')
    else:
        raise ValueError("Unsupported model type for loading labels.")

    # Get absolute path relative to the project root directory
    label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', label_path)

    logger.info(f"Looking for ground truth label at {label_path}")

    if not os.path.exists(label_path):
        logger.error(f"No such file or directory: '{label_path}'")
        raise FileNotFoundError(f"No such file or directory: '{label_path}'")

    if model_type == 'classification':
        with open(label_path, 'r') as file:
            label = int(file.read().strip())
    else:
        label_image = Image.open(label_path)
        label = np.array(label_image)

    logger.info(f"Loaded ground truth label from {label_path}")
    return label

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    logger.info("Starting upload_file function...")
    if request.method == 'POST':
        logger.info("Handling POST request...")
        
        if 'files[]' not in request.files:
            flash('No files part', 'danger')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        if not files or all(file.filename == '' for file in files):
            flash('No selected files.', 'danger')
            return redirect(request.url)

        results = []
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        for file in files:
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File saved at: {filepath}")
                
                try:
                    # Preprocess image
                    image_tensor = preprocess_image(filepath)
                    logger.info(f"Image tensor shape: {image_tensor.shape}")
                    
                    # Perform prediction
                    result = {'filename': filename}
                    try:
                        classification_prediction, classification_confidence = classification_predict(image_tensor)
                        result.update({
                            'classification': {
                                'type': 'classification',
                                'prediction': classification_prediction,
                                'confidence_score': classification_confidence,
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error in classification for {filename}: {e}")
                        result['classification'] = {'error': f"Classification failed for {filename}."}

                    try:
                        segmentation_prediction = segmentation_predict(image_tensor, model_type)
                        result.update({
                            'segmentation': {
                                'type': 'segmentation',
                                'prediction': segmentation_prediction,
                                'dice': None,
                                'iou': None,
                                'pixel_accuracy': None,
                                'mean_accuracy': None,
                                'mean_iou': None,
                                'bf1': None,
                            }
                        })

                        # Ground truth validation
                        try:
                            label = load_ground_truth_label(filepath, model_type)
                            preds = torch.tensor(segmentation_prediction.flatten())
                            labels = torch.tensor(label.flatten())
                            result['segmentation'].update({
                                'dice': dice_coefficient(preds, labels),
                                'iou': intersection_over_union(preds, labels),
                                'pixel_accuracy': pixel_accuracy(preds, labels),
                                'mean_accuracy': mean_accuracy(preds, labels),
                                'mean_iou': mean_iou(preds, labels),
                                'bf1': boundary_f1_score(preds, labels),
                            })

                            result['segmentation'] = {
                                'type': 'segmentation',
                                'prediction': segmentation_prediction if 'segmentation_prediction' in locals() else None,
                                'dice': None,
                                'iou': None,
                                'pixel_accuracy': None,
                                'mean_accuracy': None,
                                'mean_iou': None,
                                'bf1': None,
                                'warning': 'No ground truth found for metrics calculation.' if not label else None,
                            }

                        except FileNotFoundError:
                            logger.warning(f"No ground truth label found for {filename}")
                            result['segmentation']['warning'] = 'No ground truth found for metrics calculation.'
                    except Exception as e:
                        logger.error(f"Error in segmentation for {filename}: {e}")
                        result['segmentation'] = {'error': f"Segmentation failed for {filename}."}

                    # Append the result
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    flash(f"Error processing file {filename}.", "danger")

        # Render results page if any results were processed
        if results:
            return render_template('result.html', results=results)
        else:
            flash('No files were successfully processed.', 'danger')
            return redirect(request.url)

    # If GET request or no processing
    return render_template('upload.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    confidence_score = request.args.get('confidence_score')
    filename = request.args.get('filename')
    image_shape = request.args.get('image_shape')
    accuracy = request.args.get('accuracy')
    precision = request.args.get('precision')
    recall = request.args.get('recall')
    f1 = request.args.get('f1')
    conf_matrix = request.args.get('conf_matrix')

    
    logger.info(f'Result Page - Prediction: {prediction}, Confidence Score: {confidence_score}, Filename: {filename}, Image Shape: {image_shape}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
    return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_shape, accuracy=accuracy, precision=precision, recall=recall, f1=f1, conf_matrix=conf_matrix)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == ('POST'):
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Here you can handle the form submission, e.g., save to a database or send an email
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
