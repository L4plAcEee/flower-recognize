# 可用于毕业设计的花卉识别模型与大语言模型结合的识别系统
## 项目目录结构
```text
ROOT/
├── .idea/                # IDE配置文件夹 <git已排除>
├── data/                 # 数据集相关文件夹 <git已排除>
├── meta/                 # 元数据文件夹
├── models/               # 模型文件 <git已排除>
├── reports/              # 报告文件夹 <git已排除>
├── static/               # 静态资源文件夹 <git已排除>
├── templates/            # 模板文件夹
├── utils/                # 工具模块
├── .env                  # 环境变量配置文件 <git已排除>
├── .env.sample           # 环境变量示例文件
├── .gitignore            # Git忽略文件
├── app.py                # 主应用入口
├── README.md             # 项目说明文件
├── test.py               # 测试脚本
└── train.py              # 训练脚本
```

---
## 使用说明
### 前期准备
下载 **Oxford 102 花卉数据集** 并保存至项目根目录下的 `data/` 文件夹。  
接着，运行脚本 `utils/oxford102_structor.py` 对原始数据进行整理与结构化。  
最后，使用 `utils/split_dataset.py` 对整理后的数据集进行训练集、验证集和测试集的划分。  

### 模型训练
使用 **train.py** 进行训练，  
该代码基于 ResNet50 进行迁移学习，训练一个用于 102 类花卉图像分类的深度学习模型。  
它设置随机种子以保证可复现性，使用 `ImageDataGenerator` 进行数据增强，并构建包含全连接层和 Dropout 的模型顶层。  
通过 `EarlyStopping` 和 `ReduceLROnPlateau` 控制训练过程，  
并在训练完成后绘制训练曲线、评估测试集准确率、保存模型结构与结果，  
最终导出 `.h5` 模型。整体结构清晰、训练策略稳健，适用于小样本图像分类任务。  

### 模型评测
**test.py** 用于评估基于ResNet的花卉识别模型。  
它加载测试数据，执行模型预测，并计算混淆矩阵、分类报告、ROC与PR曲线等评估指标。   
同时生成错误样本可视化图和分类性能图，最终输出详细的HTML评估报告，便于全面分析模型在各类别上的表现与推理效率。  

### 模型展示
> 注意需要自己填充 大模型密钥

**app.py** 基于Flask框架，结合TensorFlow图像识别模型和Kimi大语言模型，实现花卉识别与自然语言描述生成。  
用户上传图片后，系统预测花卉类别，并自动生成详尽的花卉介绍及养护建议，  
适用于智能园艺与科普展示场景。  

---
## License
MIT License  
Copyright © 2025 L4place  
  
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.  

---