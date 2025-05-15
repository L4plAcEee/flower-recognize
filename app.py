import os
import json
import requests
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import dotenv

dotenv.load_dotenv()
# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Flask 应用配置
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载 TensorFlow 模型
model = load_model('models/flower_model_resnet.h5')

# 加载索引映射（英文花名 → 索引），并反转：索引 → 英文花名
LABEL_MAP_PATH = r'D:\coding\毕业设计-花卉识别\meta\index_mapping.json'
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    name_to_index = json.load(f)
index_to_name = {v: k for k, v in name_to_index.items()}

# Kimi API 配置
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"
KIMI_API_KEY = os.getenv("KIMI_API_KEY")


# 使用 Kimi LLM 生成花卉描述
def explain_with_kimi(flower_en_name: str) -> str:
    prompt = (
        f"请介绍一下{flower_en_name}这种花的基本特征、原产地、典型生长环境以及外观特征。"
        "要求简介清晰，适合给花卉识别应用展示使用，并生成相关养护建议，不要使用markdown格式，直接生成文段。"
    )
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "moonshot-v1-32k",  # 根据实际可用模型名修改
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(KIMI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"获取花卉描述时出错: {str(e)}")
        return f"无法获取{flower_en_name}的详细描述。"


# 预测并获得 Kimi 描述
def model_predict(img_path: str):
    # 加载并预处理图像 - 使用ResNet50的标准预处理
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # 使用ResNet50预处理函数
    img_array = preprocess_input(img_array)

    # 模型预测
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    en_name = index_to_name.get(predicted_index, "unknown")

    # 调用 Kimi 生成自然语言描述
    description = explain_with_kimi(en_name)
    return en_name, confidence, description


# Flask 路由：上传、预测、展示
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                pred_class, confidence, description = model_predict(filepath)
                prediction_results = {
                    'prediction': pred_class,
                    'confidence': round(confidence * 100, 2),
                    'img_path': filepath,
                    'description': description
                }
            except Exception as e:
                prediction_results = {
                    'error': f"处理图像时出错: {str(e)}"
                }

    return render_template('index.html', **prediction_results if prediction_results else {'prediction': None})


if __name__ == '__main__':
    app.run(debug=True)