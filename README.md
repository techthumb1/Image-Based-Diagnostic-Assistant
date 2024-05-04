# Image-Based-Diagnostic-Assistant

The Image Based Diagnostic Assistant is a web application that incorporates deep learning algorithms to assist in the diagnosis of diseases. Users of the application will be able to upload images directly into the application and receive a suggested diagnosis based on the image. Another feature of the application is the ability to determine best medication or methods for recovery based on the image (with the use of additional information provided by the user or doctor). 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install tensorflow
pip install keras
pip install flask
pip install numpy
pip install opencv-python
pip install flask-cors
pip install pillow
pip install flask-ngrok
```

## Usage

```python
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        prediction = model.predict(img)
        return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()

```

## Contributing