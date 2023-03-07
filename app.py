from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load pre-trained InceptionV3 model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Define list of plant diseases
plant_diseases = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

# Define function to preprocess image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    # Resize image
    img = cv2.resize(img, (299, 299))
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to numpy array
    img = np.array(img)
    # Preprocess input
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# Define function to predict plant disease
def predict_plant_disease(image_path):
    # Preprocess image
    img = preprocess_image(image_path)
    # Make prediction using model
    preds = model.predict(np.array([img]))
    # Decode prediction
    decoded_preds = tf.keras.applications.inception_v3.decode_predictions(preds, top=1)[0][0]
    # Get predicted class name
    predicted_class = decoded_preds[1]
    return predicted_class

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Define route for plant health analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get image from form
    image_file = request.files['image']
    # Save image to temporary folder
    filename = secure_filename(image_file.filename)
    image_path = f'temp/{filename}'
    image_file.save(image_path)
    # Predict plant disease
    predicted_class = predict_plant_disease(image_path)
    # Return result
    result = {
        'predicted_class': predicted_class
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
