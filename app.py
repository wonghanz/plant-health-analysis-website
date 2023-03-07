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
