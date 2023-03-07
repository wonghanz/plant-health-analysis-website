import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Initialize Flask app
app = Flask(__name__)

# Set path for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit uploaded file size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Define function to analyze plant health
def analyze_plant_health(image_path):
    # Load image and preprocess
    image = load_img(image_path, target_size=(299, 299))
    x = img_to_array(image)
    x = preprocess_input(x)

    # Use model to predict class
    predictions = model.predict(x.reshape(1, 299, 299, 3))
    predicted_class = decode_predictions(predictions, top=1)[0][0]

    # Return predicted class and probability
    return predicted_class[1], round(predicted_class[2] * 100, 2)

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is uploaded
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # Check if file is valid image
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)

    # Save file to UPLOAD_FOLDER
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Analyze plant health and return results
    plant_class, probability = analyze_plant_health(file_path)

    # Render results template with marks
    if plant_class == 'healthy':
        mark = '✔️'
    else:
        mark = '❌'

    return render_template('results.html', filename=filename, plant_class=plant_class, probability=probability, mark=mark)

if __name__ == '__main__':
    app.run(debug=True)
