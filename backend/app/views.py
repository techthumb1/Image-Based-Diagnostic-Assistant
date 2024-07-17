import os
import numpy as np
import logging
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
from utils.config import Config, load_yaml_config, apply_config
from app.utils import allowed_file
from models.model_training import get_classification_model, get_unetpp_model, get_efficientnet_model
from models.metrics import (
    calculate_metrics,
    dice_coefficient,
    intersection_over_union,
    pixel_accuracy,
    mean_accuracy,
    mean_iou,
    boundary_f1_score
    #compute_accuracy,
    #compute_precision,
    #compute_recall,
    #compute_f1_score,
    #compute_auc_roc,
    #compute_confusion_matrix
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from huggingface_hub import login
from safetensors.torch import load_file
from models.data_loader import transform
from models.unetpp import UNetPP, load_and_preprocess_image, model_predict
from models.data_loader import preprocess_image
from efficientnet_pytorch import EfficientNet

import sys
#sys.setrecursionlimit(2000)  # Temporarily increase the recursion limit

# Global Variables
classification_model = None
segmentation_model = None
unetpp_model = None
feature_extractor = None

# Ignore FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

app = Flask(__name__)

config = load_yaml_config('config/config.yaml')
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

# Load model and feature extractor based on the configuration
pruned_model_path = app.config.get('PRUNED_MODEL_PATH')
feature_extractor_path = 'nvidia/segformer-b0-finetuned-ade-512-512'
model_type = app.config.get('MODEL_TYPE', 'classification')
#feature_extractor_path = app.config.get('FEATURE_EXTRACTOR_PATH')

# Load the appropriate model and feature extractor based on model_type
model_dir = app.config.get('MODEL_DIR', 'models/segformer_pruned_model')
config_path = os.path.join(model_dir, 'config.json')
model_path = os.path.join(model_dir, 'model.safetensors')
efficientnet_model_path = 'backend/models/effnet_classification_model_best.pth'  # Update with the actual path
unetpp_model_path = 'backend/models/unetpp_model.pth'  # Update with the actual path

model_type = app.config.get('MODEL_TYPE', 'classification')

# Load the appropriate model and feature extractor based on model_type
try:
    if model_type == 'classification':
        logger.info("Loading the EfficientNet model for classification...")
        classification_model = EfficientNet.from_name('efficientnet-b0')  # Adjust the architecture name as needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classification_model.to(device)
        #classification_model = get_classification_model(num_classes=3)
        #classification_model.load_state_dict(torch.load(efficientnet_model_path, map_location=torch.device('cpu')))
        #classification_model.to(torch.device('cpu'))
        logger.info("EfficientNet model loaded successfully")
    elif model_type == 'segmentation':
        logger.info("Loading the Segformer model for segmentation...")

        config = SegformerConfig.from_pretrained(config_path)
        state_dict = load_file(model_path)
        segmentation_model = SegformerForSemanticSegmentation(config)
        segmentation_model.load_state_dict(state_dict)

        logger.info("Segmentation model loaded successfully")
    elif model_type == 'unetpp':
        logger.info("Loading the UNet++ model...")
        unetpp_model = get_unetpp_model()
        unetpp_model.load_state_dict(torch.load(unetpp_model_path, map_location=torch.device('cpu')))
        logger.info("UNet++ model loaded successfully")
    else:
        raise ValueError("Unsupported model type specified in configuration.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    logger.info("Loading the feature extractor...")
    feature_extractor = SegformerImageProcessor.from_pretrained(config_path)
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
    with torch.no_grad():
        outputs = classification_model(inputs)
    
    # Handle outputs directly
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    
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













# Inside your upload_file function
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    logger.info("Starting upload_file function...")
    if request.method == 'POST':
        logger.info("Handling POST request...")
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename, app.config['uploads']['allowed_extensions']):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['uploads']['folder'], filename)

            # Ensure the uploads directory exists
            if not os.path.exists(app.config['uploads']['folder']):
                os.makedirs(app.config['uploads']['folder'])

            file.save(filepath)
            logger.info(f"File saved at: {filepath}")

            try:
                image_tensor = preprocess_image(filepath)
                logger.info(f"Image tensor shape: {image_tensor.shape}")
                if image_tensor is not None:
                    if model_type == 'classification':
                        prediction, confidence_score = classification_predict(image_tensor)
                    else:
                        prediction = segmentation_predict(image_tensor, model_type)
                        confidence_score = None

                    logger.info(f'Prediction: {prediction}, Confidence Score: {confidence_score}, Filename: {filename}, Image Shape: {image_tensor.shape}')

                    # Load ground truth label
                    labels_folder = app.config['labels']['folder']
                    label_filename = f"{os.path.splitext(filename)[0]}_label.txt"
                    label_path = os.path.abspath(os.path.join(labels_folder, label_filename))
                    logger.info(f"Looking for ground truth label at {label_path}")
                    if os.path.exists(label_path):
                        label = load_ground_truth_label(label_path, model_type)
                        logger.info(f"Loaded ground truth label from {label_path} with shape: {label.shape}")

                        # Ensure labels and predictions have the correct dimensions
                        if model_type in ['segmentation', 'unetpp']:
                            if isinstance(label, np.ndarray):
                                label = torch.tensor(label)
                            elif isinstance(label, int):
                                label = torch.tensor([label])

                            if label.ndim == 2:
                                label = label.unsqueeze(0).unsqueeze(0)
                            elif label.ndim == 3:
                                label = label.unsqueeze(1)

                            if prediction.ndim == 3:
                                prediction = prediction.unsqueeze(0)

                            logger.info(f"Prediction shape: {prediction.shape}, Label shape: {label.shape}")

                            preds = prediction.flatten()
                            labels = label.flatten()

                            logger.debug("Starting metric calculations...")
                            dice_score = dice_coefficient(preds, labels)
                            iou_score = intersection_over_union(preds, labels)
                            pixel_acc = pixel_accuracy(preds, labels)
                            mean_acc = mean_accuracy(preds, labels)
                            mean_iou_val = mean_iou(preds, labels)
                            bf1 = boundary_f1_score(preds, labels)

                            acc, prec, rec, f1, auc, avg_prec, conf_matrix = calculate_metrics(labels.cpu().numpy(), preds.cpu().numpy())
                            flash(f'Dice Coefficient: {dice_score}, IoU: {iou_score}, Pixel Accuracy: {pixel_acc}, Mean Accuracy: {mean_acc}, Mean IoU: {mean_iou_val}, BF1: {bf1}', 'success')
                            return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_tensor.shape, dice=dice_score, iou=iou_score, pixel_accuracy=pixel_acc, mean_accuracy=mean_acc, mean_iou=mean_iou_val, bf1=bf1)
                        else:
                            preds = torch.tensor([prediction])
                            labels = torch.tensor([label])

                            logger.info(f"Prediction shape: {preds.shape}, Label shape: {labels.shape}")

                            acc, prec, rec, f1, auc, avg_prec, conf_matrix = calculate_metrics(labels.cpu().numpy(), preds.cpu().numpy())

                            flash(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}, AUC-ROC: {auc}', 'success')
                            return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_tensor.shape, accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc, conf_matrix=conf_matrix)
                    else:
                        logger.warning(f"No ground truth label found for {filename}. Skipping metric calculation.")

                        flash(f'Prediction: {prediction}, Confidence Score: {confidence_score}', 'success')
                        return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_tensor.shape)
                else:
                    flash('Failed to preprocess image.', 'danger')
                    return redirect(request.url)
            except UnidentifiedImageError as e:
                logger.error(f"Unsupported image format: {e}")
                flash('Unsupported image format. Please upload a JPEG, PNG, GIF, or BMP file.', 'danger')
                return redirect(request.url)
            except KeyError as e:
                logger.error(f"Configuration key error: {e}")
                flash('Configuration key error.', 'danger')
                return redirect(request.url)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                flash('An error occurred while processing the file.', 'danger')
                return redirect(request.url)
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
