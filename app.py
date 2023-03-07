import os
from flask import Flask, request, render_template
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# 加载预训练的 InceptionV3 模型
model = InceptionV3(weights='imagenet')

# 设置上传文件夹
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 从 HTML 表单中获取文件并保存
        file = request.files['file']
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(fpath)

        # 加载图像并进行预处理
        image = load_img(fpath, target_size=(299, 299))
        x = img_to_array(image)
        x = preprocess_input(x)

        # 使用模型进行预测
        predictions = model.predict(x.reshape(1, 299, 299, 3))

        # 解码预测结果
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=3)[0]

        # 渲染结果页面
        return render_template('result.html', predictions=decoded_predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
