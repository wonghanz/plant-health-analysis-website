from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('plant_health_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0]
    if prediction[0] > prediction[1]:
        result = 'Healthy'
    else:
        result = 'Unhealthy'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Load plant image and resize it to 299x299
img_path = 'plant.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (299, 299))

# Convert the image to a numpy array and preprocess it for InceptionV3 model
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Use the model to predict the class of the image
predictions = model.predict(x)

# Decode the prediction result
predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0][0]

# Get the predicted class label and its confidence score
label = predicted_class[1]
score = predicted_class[2]

# Set a threshold for classifying healthy and unhealthy plants
threshold = 0.5

# Determine the health status of the plant based on the prediction score
if score >= threshold:
    health_status = "healthy"
else:
    health_status = "unhealthy"

# Draw a label on the image indicating the health status of the plant
cv2.putText(img, health_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the image with the health status label
cv2.imshow("Plant Health Analysis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
