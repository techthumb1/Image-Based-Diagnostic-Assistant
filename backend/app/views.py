from flask import render_template, request
from model.cnn_model import load_model, model_predict
import os

app = Flask(__name__)
model = load_model('models/path_to_model/model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        prediction = model_predict(filename)
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')
