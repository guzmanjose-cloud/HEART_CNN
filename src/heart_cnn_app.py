from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.utils import to_categorical
import joblib

# Define constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
disease_mapping = {0: 'coronary artery disease', 1: 'angina disease', 2: 'hypotension disease', 3: 'cardio vascular disease'}

# Load the saved model and KMeans model
model = tf.keras.models.load_model('/workspaces/HEART_CNN/heart_disease_model.h5')
kmeans = joblib.load('/workspaces/HEART_CNN/kmeans_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Helper function to process the image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize(IMG_SIZE)
    # Save the image temporarily as a .jpe file
    temp_path = '/tmp/temp_image.jpe'
    img.save(temp_path, format='JPEG')

    # Reload the image as a .jpe
    img_jpe = Image.open(temp_path)
    img_array = np.array(img_jpe) / 255.0
    print("type:", img_jpe)
    print("path:", temp_path)

    #img_array = np.array(img) / 255.0
    img_flattened = img_array.reshape(1, -1)
    cluster_label = kmeans.predict(img_flattened)
    cluster_label_one_hot = to_categorical(cluster_label, num_classes=NUM_CLASSES)
    cluster_label_one_hot_expanded = np.tile(cluster_label_one_hot[:, np.newaxis, np.newaxis, :], (1, IMG_SIZE[0], IMG_SIZE[1], 1))
    img_with_cluster = np.concatenate([img_array[np.newaxis, :, :, :], cluster_label_one_hot_expanded], axis=-1)
    return img_with_cluster

# Route for the main page
@app.route('/')
def index():
    return render_template('template.html')

# Route to handle image upload and prediction for multiple images
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No images uploaded'})

    files = request.files.getlist('image')
    if len(files) > 10:
        return jsonify({'error': 'Please upload a maximum of 10 images'})

    predictions = []
    for file in files:
        processed_image = preprocess_image(file)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = disease_mapping[predicted_class]
        predictions.append({'filename': file.filename, 'prediction': result})
    
    return jsonify(predictions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
