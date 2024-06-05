import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
#from models.cnn_model import load_model, model_predict
from utils.config import load_config
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load configuration
config = load_config('config/config.yaml')
app.config['UPLOAD_FOLDER'] = config['uploads']['folder']
app.config['ALLOWED_EXTENSIONS'] = set(config['uploads']['allowed_extensions'])
app.secret_key = config['app']['secret_key']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load model and configuration
model_path = 'models/path_to_model/model.h5'
config_path = 'config/config.yaml'


try:
    model = load_model(model_path)
    config = load_config(config_path)
    logger.info("Model and configuration loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or configuration: {e}")
    raise


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


app.route('/')
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
        if file:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            try:
                prediction = model_predict(filename)
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
