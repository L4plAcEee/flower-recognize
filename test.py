import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import pandas as pd
import time

# 设置中文支持
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置相关路径
PATH = './data/split_flowers'
MODEL_PATH = 'models/flower_model_resnet.h5'
REPORT_DIR = './reports/evaluation'
os.makedirs(REPORT_DIR, exist_ok=True)

# 基本参数设置
batch_size = 32
input_shape = (224, 224)


def load_and_prepare_data():
    """
    加载和准备测试数据集
    """
    # 创建ImageDataGenerator用于测试集，使用相同的预处理
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # 加载测试数据
    test_generator = test_datagen.flow_from_directory(
        os.path.join(PATH, 'test'),
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # 保持顺序不变，对应类别标签
    )

    # 获取类别映射
    class_indices = test_generator.class_indices
    classes = {v: k for k, v in class_indices.items()}

    return test_generator, classes


def evaluate_model(model, test_generator):
    """
    评估模型并返回预测结果和真实标签
    """
    # 使用模型进行预测
    start_time = time.time()
    y_pred_probs = model.predict(test_generator)
    inference_time = time.time() - start_time

    # 计算每个样本的预测时间
    samples_count = len(test_generator.filenames)
    avg_inference_time = inference_time / samples_count

    # 获取预测类别
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 获取真实标签
    y_true = test_generator.classes

    return y_pred_probs, y_pred, y_true, samples_count, avg_inference_time


def calculate_metrics(y_true, y_pred, y_pred_probs, classes):
    """
    计算各种评估指标
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算各类别和总体的准确率、精确率、召回率、F1值等
    report = classification_report(y_true, y_pred, target_names=list(classes.values()), output_dict=True)

    # 计算每个类别的特定指标
    class_metrics = {}
    for class_idx in range(len(classes)):
        class_name = classes[class_idx]

        # 将当前类视为正类，其他为负类（一对多）
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_prob = y_pred_probs[:, class_idx]

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_prob)

        class_metrics[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall
        }

    return report, cm, class_metrics


def plot_confusion_matrix(cm, classes, normalize=False):
    """
    绘制混淆矩阵
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "归一化混淆矩阵"
    else:
        title = "混淆矩阵"

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='.2f' if normalize else 'd')
    plt.title(title, fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f'confusion_matrix_{"norm" if normalize else "raw"}.png'))
    plt.close()


def plot_roc_curves(class_metrics, classes, num_classes=10):
    """
    绘制ROC曲线（仅展示部分类别）
    """
    plt.figure(figsize=(10, 8))

    # 选择前num_classes个类别进行可视化
    selected_classes = list(class_metrics.keys())[:num_classes]

    for class_name in selected_classes:
        metrics = class_metrics[class_name]
        plt.plot(metrics['fpr'], metrics['tpr'],
                 lw=2, label=f'{class_name} (AUC = {metrics["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('花卉识别模型 ROC 曲线')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(REPORT_DIR, 'roc_curves.png'))
    plt.close()


def plot_precision_recall_curves(class_metrics, classes, num_classes=10):
    """
    绘制PR曲线（仅展示部分类别）
    """
    plt.figure(figsize=(10, 8))

    # 选择前num_classes个类别进行可视化
    selected_classes = list(class_metrics.keys())[:num_classes]

    for class_name in selected_classes:
        metrics = class_metrics[class_name]
        plt.plot(metrics['recall'], metrics['precision'],
                 lw=2, label=f'{class_name}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('花卉识别模型 精确率-召回率 曲线')
    plt.legend(loc="lower left")

    plt.savefig(os.path.join(REPORT_DIR, 'pr_curves.png'))
    plt.close()


def plot_error_samples(model, test_generator, classes, num_samples=10):
    """
    可视化一些错误预测的样本
    """
    # 获取所有预测和真实标签
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # 找出错误预测的样本
    misclassified_indices = np.where(y_pred != y_true)[0]

    if len(misclassified_indices) == 0:
        print("没有错误分类的样本")
        return

    # 只选择前num_samples个样本
    if len(misclassified_indices) > num_samples:
        misclassified_indices = misclassified_indices[:num_samples]

    # 获取所有批次的数据
    all_images = []
    all_labels = []

    test_generator.reset()
    for i in range(len(test_generator)):
        images, labels = test_generator.next()
        all_images.append(images)
        all_labels.append(labels)

        if i * batch_size + len(images) >= len(test_generator.filenames):
            break

    all_images = np.vstack([img for img in all_images if len(img) > 0])

    # 绘制错误预测的样本
    plt.figure(figsize=(20, 4 * ((num_samples + 4) // 5)))

    for i, idx in enumerate(misclassified_indices):
        plt.subplot(((num_samples + 4) // 5), 5, i + 1)

        # 反归一化图像以便可视化
        img = all_images[idx]
        # 从预处理的图像恢复
        img = img.copy()
        img /= 2.0
        img += 0.5
        img *= 255.0
        img = np.clip(img, 0, 255).astype('uint8')

        plt.imshow(img)
        true_class = classes[y_true[idx]]
        pred_class = classes[y_pred[idx]]
        plt.title(f"真实: {true_class}\n预测: {pred_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'misclassified_samples.png'))
    plt.close()


def analyze_per_class_performance(report, classes):
    """
    分析每个类别的性能并生成报告
    """
    # 创建一个DataFrame来存储每个类别的性能指标
    class_names = []
    precision = []
    recall = []
    f1_score = []
    support = []

    for class_name in classes.values():
        if class_name in report:
            class_names.append(class_name)
            precision.append(report[class_name]['precision'])
            recall.append(report[class_name]['recall'])
            f1_score.append(report[class_name]['f1-score'])
            support.append(report[class_name]['support'])

    # 创建DataFrame
    df = pd.DataFrame({
        '类别': class_names,
        '精确率': precision,
        '召回率': recall,
        'F1值': f1_score,
        '样本数': support
    })

    # 按F1值排序
    df = df.sort_values(by='F1值', ascending=False)

    # 保存到CSV
    df.to_csv(os.path.join(REPORT_DIR, 'per_class_performance.csv'), index=False, encoding='utf-8-sig')

    # 绘制前20个和后20个类别的性能对比
    top_classes = df.head(20)
    bottom_classes = df.tail(20)

    # 绘制前20个类别的性能
    plt.figure(figsize=(15, 8))
    x = np.arange(len(top_classes))
    width = 0.2

    plt.bar(x - width, top_classes['精确率'], width, label='精确率')
    plt.bar(x, top_classes['召回率'], width, label='召回率')
    plt.bar(x + width, top_classes['F1值'], width, label='F1值')

    plt.xlabel('花卉类别')
    plt.ylabel('指标值')
    plt.title('性能最佳的20个花卉类别')
    plt.xticks(x, top_classes['类别'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'top_performing_classes.png'))
    plt.close()

    # 绘制后20个类别的性能
    plt.figure(figsize=(15, 8))
    x = np.arange(len(bottom_classes))

    plt.bar(x - width, bottom_classes['精确率'], width, label='精确率')
    plt.bar(x, bottom_classes['召回率'], width, label='召回率')
    plt.bar(x + width, bottom_classes['F1值'], width, label='F1值')

    plt.xlabel('花卉类别')
    plt.ylabel('指标值')
    plt.title('性能最差的20个花卉类别')
    plt.xticks(x, bottom_classes['类别'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'bottom_performing_classes.png'))
    plt.close()

    # 返回分析结果
    return df


def generate_html_report(report, model_metrics, sample_count, inference_time):
    """
    生成HTML格式的评估报告
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>花卉识别模型评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .metric-box {{ flex: 1; margin: 10px; padding: 15px; border-radius: 5px; background-color: #f5f5f5; min-width: 200px; }}
            h1, h2, h3 {{ color: #333; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>花卉识别模型评估报告</h1>
            <p>评估日期: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>模型性能总览</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>准确率</h3>
                    <p>{report['accuracy']:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>加权精确率</h3>
                    <p>{report['weighted avg']['precision']:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>加权召回率</h3>
                    <p>{report['weighted avg']['recall']:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>加权F1值</h3>
                    <p>{report['weighted avg']['f1-score']:.4f}</p>
                </div>
            </div>

            <h2>推理性能</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>测试样本数</h3>
                    <p>{sample_count}</p>
                </div>
                <div class="metric-box">
                    <h3>总推理时间</h3>
                    <p>{inference_time:.2f} 秒</p>
                </div>
                <div class="metric-box">
                    <h3>平均每样本推理时间</h3>
                    <p>{(inference_time / sample_count) * 1000:.2f} 毫秒</p>
                </div>
            </div>

            <h2>评估结果可视化</h2>
            <div class="figure">
                <h3>混淆矩阵</h3>
                <img src="confusion_matrix_norm.png" alt="归一化混淆矩阵">
            </div>

            <div class="figure">
                <h3>ROC曲线</h3>
                <img src="roc_curves.png" alt="ROC曲线">
            </div>

            <div class="figure">
                <h3>精确率-召回率曲线</h3>
                <img src="pr_curves.png" alt="精确率-召回率曲线">
            </div>

            <div class="figure">
                <h3>性能最佳的类别</h3>
                <img src="top_performing_classes.png" alt="性能最佳的类别">
            </div>

            <div class="figure">
                <h3>性能最差的类别</h3>
                <img src="bottom_performing_classes.png" alt="性能最差的类别">
            </div>

            <div class="figure">
                <h3>错误分类样本</h3>
                <img src="misclassified_samples.png" alt="错误分类样本">
            </div>
        </div>
    </body>
    </html>
    """

    with open(os.path.join(REPORT_DIR, 'evaluation_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """
    主函数，执行完整的评估流程
    """
    print("开始花卉识别模型评估...")

    # 加载测试数据
    test_generator, classes = load_and_prepare_data()
    print(f"加载测试数据完成，共{len(test_generator.filenames)}个样本，{len(classes)}个类别")

    # 加载模型
    model = load_model(MODEL_PATH)
    print(f"模型已从{MODEL_PATH}加载")

    # 评估模型
    y_pred_probs, y_pred, y_true, sample_count, avg_inference_time = evaluate_model(model, test_generator)
    print(f"模型评估完成，平均每样本推理时间：{avg_inference_time * 1000:.2f}毫秒")

    # 计算各种指标
    report, cm, class_metrics = calculate_metrics(y_true, y_pred, y_pred_probs, classes)
    print(f"测试集准确率: {report['accuracy']:.4f}")
    print(f"加权精确率: {report['weighted avg']['precision']:.4f}")
    print(f"加权召回率: {report['weighted avg']['recall']:.4f}")
    print(f"加权F1值: {report['weighted avg']['f1-score']:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, classes.values(), normalize=False)
    plot_confusion_matrix(cm, classes.values(), normalize=True)
    print("混淆矩阵已绘制")

    # 绘制ROC曲线
    plot_roc_curves(class_metrics, classes)
    print("ROC曲线已绘制")

    # 绘制PR曲线
    plot_precision_recall_curves(class_metrics, classes)
    print("PR曲线已绘制")

    # 分析每个类别的性能
    class_performance = analyze_per_class_performance(report, classes)
    print("各类别性能分析完成")

    # 可视化错误样本
    plot_error_samples(model, test_generator, classes)
    print("错误样本可视化完成")

    # 生成HTML报告
    total_inference_time = avg_inference_time * sample_count
    generate_html_report(report, class_metrics, sample_count, total_inference_time)
    print(f"HTML评估报告已生成至{os.path.join(REPORT_DIR, 'evaluation_report.html')}")

    # 保存总体指标到CSV
    metrics_df = pd.DataFrame({
        '指标': ['准确率', '加权精确率', '加权召回率', '加权F1值', '测试样本数', '总推理时间(秒)',
                 '平均每样本推理时间(毫秒)'],
        '值': [
            report['accuracy'],
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score'],
            sample_count,
            total_inference_time,
            avg_inference_time * 1000
        ]
    })
    metrics_df.to_csv(os.path.join(REPORT_DIR, 'overall_metrics.csv'), index=False, encoding='utf-8-sig')

    print("评估完成！")


if __name__ == "__main__":
    main()