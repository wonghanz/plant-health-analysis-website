from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# 加载预训练的 InceptionV3 模型
model = InceptionV3(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']

        # 加载植物的照片
        image = load_img(file, target_size=(299, 299))

        # 将照片转换为 NumPy 数组，并进行预处理
        x = img_to_array(image)
        x = preprocess_input(x)

        # 使用模型进行预测
        predictions = model.predict(tf.expand_dims(x, axis=0))

        # 解码预测结果
        predicted_class = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0][0]

        # 根据预测结果返回相应的页面
        if predicted_class[1] == 'healthy':
            return render_template('healthy.html')
        else:
            return render_template('unhealthy.html')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
