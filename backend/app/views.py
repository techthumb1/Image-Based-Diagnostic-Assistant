import os
import numpy as np
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from transformers import SegformerForImageClassification, SegformerImageProcessor
from PIL import UnidentifiedImageError
import torch
from utils.config import Config, load_yaml_config, apply_config
from app.utils import load_and_preprocess_image, allowed_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_sqlalchemy import SQLAlchemy
from huggingface_hub import login

app = Flask(__name__)
apply_config(app)

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['ALLOWED_EXTENSIONS'] = set(os.getenv('ALLOWED_EXTENSIONS').split(','))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY')
app.debug = os.getenv('DEBUG', 'false').lower() in ['true', '1']

db = SQLAlchemy(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
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
login(token=hf_token)

# Load model and feature extractor
pruned_model_path = app.config.get('PRUNED_MODEL_PATH')
feature_extractor_path = app.config.get('FEATURE_EXTRACTOR_PATH')

try:
    model = SegformerForImageClassification.from_pretrained(pruned_model_path, use_auth_token=hf_token)
    feature_extractor = SegformerImageProcessor.from_pretrained(feature_extractor_path, use_auth_token=hf_token)
    logger.info("Model and feature extractor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or feature extractor: {e}")
    raise

def model_predict(image_array):
    inputs = feature_extractor(images=image_array, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    logger.info(f'Predicted class index: {predicted_class_idx}')
    return predicted_class_idx

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

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the uploads directory exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(filepath)
            try:
                logger.info(f"File saved at: {filepath}")
                image_array = load_and_preprocess_image(filepath)
                if image_array is not None:
                    prediction = model_predict(image_array)
                    confidence_score = np.max(torch.nn.functional.softmax(model(**feature_extractor(images=image_array, return_tensors="pt")).logits.detach(), dim=-1).numpy())
                    logger.info(f'Prediction: {prediction}, Confidence Score: {confidence_score}, Filename: {filename}, Image Shape: {image_array.shape}')
                    flash(f'Prediction successful: {prediction} with confidence score {confidence_score:.2f}', 'success')
                    return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_array.shape)
                else:
                    flash('Failed to preprocess image.', 'danger')
                    return redirect(request.url)
            except UnidentifiedImageError as e:
                logger.error(f"Unsupported image format: {e}")
                flash('Unsupported image format. Please upload a JPEG, PNG, GIF, or BMP file.', 'danger')
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
    logger.info(f'Result Page - Prediction: {prediction}, Confidence Score: {confidence_score}, Filename: {filename}, Image Shape: {image_shape}')
    return render_template('result.html', prediction=prediction, confidence_score=confidence_score, filename=filename, image_shape=image_shape)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
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
