import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import json

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载预训练的 InceptionV3 模型，并微调以用于植物健康分析
def load_inception_model():
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('inception_model.h5')
    return model

inception_model = load_inception_model()

# 检查文件类型是否符合要求
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 预处理并分析植物的照片
def analyze_plant(filename):
    image = load_img(filename, target_size=(299, 299))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = inception_model.predict(x)
    return np.argmax(prediction[0])

# 网站首页
@app.route('/')
def index():
    return render_template('index.html')

# 处理文件上传请求
@app.route('/analyze', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = analyze_plant(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'result': result})
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
