import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import SegformerForImageClassification, SegformerImageProcessor
from PIL import Image
import torch
from utils.config import Config, load_yaml_config
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and feature extractor
model_path = app.config['MODEL_PATH']
feature_extractor_path = app.config['MODEL_PATH']  # Assuming they are in the same path

try:
    model = SegformerForImageClassification.from_pretrained(model_path)
    feature_extractor = SegformerImageProcessor.from_pretrained(feature_extractor_path)
    logger.info("Model and feature extractor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or feature extractor: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def model_predict(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                prediction = model_predict(filepath)
                return render_template('result.html', prediction=prediction)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                flash('An error occurred while processing the file.')
                return redirect(request.url)
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
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
