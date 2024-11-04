from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import resize_with_pad
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

model = load_model('best_transfer_cnn.h5')

with open('label_names.json', 'r') as f:
    label_names = json.load(f)

def preprocess_image(image_path, target_height, target_width):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = resize_with_pad(image, target_height, target_width, antialias=True)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image_path):
    image = preprocess_image(image_path, 224, 224)
    prediction = model.predict(image)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label_name = label_names[predicted_label_index]
    return predicted_label_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error', 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_label = predict_image(filepath)
        os.remove(filepath)
        return jsonify({'label': predicted_label})

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)