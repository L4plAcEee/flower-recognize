import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ——————————————
# 设置随机种子，以保证多次运行结果可复现
# ——————————————
seed = random.randint(0, 2**31-1)
print(f"当前种子: {seed} \n")
tf.random.set_seed(seed)

# ——————————————
# 路径和超参数配置
# ——————————————
PATH = './data/split_flowers'
MODEL_PATH = 'models/flower_model_resnet.h5'
REPORT_DIR = './reports'

input_shape = (224, 224, 3)
num_classes = 102
dropout_rate = 0.5
learning_rate = 0.0001  # 降低初始学习率
batch_size = 32

# ——————————————
# 数据预处理与增强 - ResNet50专用预处理
# ——————————————
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # 使用ResNet专用预处理
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 验证集和测试集也需要使用相同的预处理函数
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ——————————————
# 从目录中生成批量数据
# ——————————————
train_generator = train_datagen.flow_from_directory(
    os.path.join(PATH, 'train'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(PATH, 'val'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(PATH, 'test'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ——————————————
# 创建基于迁移学习的模型 - 修改架构
# ——————————————
# 加载预训练的ResNet50模型（不包含顶层分类器）
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# 解冻部分底层 - 尝试解冻更多层
for layer in base_model.layers[:-100]:  # 冻结前面的层，后面100层可训练
    layer.trainable = False
for layer in base_model.layers[-100:]:  # 解冻后面100层
    layer.trainable = True

# 构建完整模型 - 使用更强大的顶层分类器
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),  # 增加神经元数量
    Dropout(0.3),  # 减少dropout
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# ——————————————
# 编译模型 - 使用较小的学习率
# ——————————————
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ——————————————
# 配置回调函数
# ——————————————
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,        # 更激进的学习率衰减
    patience=3,        # 更早开始衰减
    min_lr=1e-6,
    verbose=1
)

# ——————————————
# 一次性训练模型 - 不再分两个阶段
# ——————————————
print("\n开始训练迁移学习模型")
history = model.fit(
    train_generator,
    epochs=40,  # 增加轮数
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)
# ——————————————
# 定义绘制训练/验证曲线函数
# ——————————————
def plot_training_curves(history, save_path=None):
    os.makedirs(REPORT_DIR, exist_ok=True)  # 确保报告目录存在

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # 可选：保存为图片
    plt.show()


# ——————————————
# 保存并展示训练曲线
# ——————————————
plot_training_curves(history, save_path=os.path.join(REPORT_DIR, 'training_curves_resnet.png'))

# ——————————————
# 将模型结构摘要保存到文本文件
# ——————————————
with open(os.path.join(REPORT_DIR, 'model_summary_resnet.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# ——————————————
# 在测试集上评估最终模型性能
# ——————————————
test_loss, test_acc = model.evaluate(test_generator)
print(f'测试集准确率: {test_acc:.4f}')

# ——————————————
# 保存测试准确率到报告
# ——————————————
with open(os.path.join(REPORT_DIR, 'test_result_resnet.txt'), 'w') as f:
    f.write(f'测试集准确率: {test_acc:.4f}\n')

# ——————————————
# 最后将训练好的模型保存到指定路径
# ——————————————
model.save(MODEL_PATH)
print(f"模型已保存到: {MODEL_PATH}")