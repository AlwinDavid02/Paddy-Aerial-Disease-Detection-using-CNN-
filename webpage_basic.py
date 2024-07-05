from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = r'Z:\Final Year Project\App\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model(r'Z:\Final Year Project\New folder\mango1.h5')  # Update with the correct filename
input_shape = (64, 64, 3)  # Adjust based on your image dimensions
class_labels = ['Brownspot', 'Healthy', 'Hispa', 'LeafBlast']  # Replace with your actual class labels

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=input_shape[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_disease_name = class_labels[predicted_class_index]
    return predicted_disease_name, predictions.flatten().tolist()

def get_preventive_remedies(disease_name):
    # Assuming separate text files for each disease category
    file_path = rf'Z:\Final Year Project\remedies\{disease_name}\New Text Document.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            remedies = file.read()
        return remedies
    else:
        return "No remedies found."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Generate a unique filename for each uploaded image
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    img_array = preprocess_image(img_path)
    result, prediction_values = predict_disease(img_array)
    remedies = get_preventive_remedies(result)

    return jsonify({
        'result': result,
        'image_path': f'/uploads/{filename}',
        'remedies': remedies,
        'prediction_values': prediction_values
    })

if __name__ == '__main__':
    app.run(debug=True)
