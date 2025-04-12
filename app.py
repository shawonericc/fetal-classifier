from flask import Flask, request, render_template, send_from_directory
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
MODEL_PATH = 'best_model_densenet169.h5'
MODEL_DRIVE_ID = '101GDngJmVrhIj81K_9lwpv9fMLB0hK_n'
DRIVE_URL = f'https://drive.google.com/uc?export=download&id={MODEL_DRIVE_ID}'

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    with requests.get(DRIVE_URL, stream=True) as r:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Model downloaded.")

model = load_model(MODEL_PATH)
class_labels = np.load('class_labels.npy', allow_pickle=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INPUT_SIZE = 224

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = np.stack([img] * 3, axis=-1)
    img = img.astype('float32') / 255.0
    img = (img - 0.5) * 2
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist('images')
        for file in uploaded_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = preprocess(filepath)
            pred = model.predict(img)
            pred_class = class_labels[np.argmax(pred)]
            predictions.append((file.filename, pred_class))
    return render_template('index.html', predictions=predictions)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
