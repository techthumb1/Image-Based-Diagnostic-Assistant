import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from utils.config import load_config
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config.from_object(Config)

# Load configuration
#config = load_config('config/config.yaml')
#app.config['UPLOAD_FOLDER'] = config['uploads']['folder']
#app.config['ALLOWED_EXTENSIONS'] = set(config['uploads']['allowed_extensions'])
#app.secret_key = config['app']['secret_key']
#app.debug = config['app']['debug']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model_path = config['model']['path']

try:
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                prediction, confidence = model_predict(filepath, model)
                return render_template('result.html', prediction=prediction, confidence=confidence)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                flash('An error occurred while processing the file.')
                return redirect(request.url)
        else:
            flash('File type not allowed')
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

def model_predict(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['Benign', 'Malignant']  # Adjust based on your classes
    return class_names[tf.argmax(score)], 100 * tf.reduce_max(score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
