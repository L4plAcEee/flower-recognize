import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

source_dir = "D:/coding/毕业设计-花卉识别/data/flowers"  # 原始数据集路径（按类分子文件夹）
output_dir = "/data/split_flowers"  # 输出路径
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15  # 数据划分比例

def split_dataset(source_dir, output_dir, train_ratio, val_ratio, test_ratio):
    """从源目录提取图片并按比例划分为训练集、验证集和测试集"""
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"发现以下类别: {class_names}")

    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        class_path = os.path.join(source_dir, class_name)
        image_paths = glob.glob(os.path.join(class_path, '*.jpg')) + \
                      glob.glob(os.path.join(class_path, '*.jpeg')) + \
                      glob.glob(os.path.join(class_path, '*.png'))

        if not image_paths:
            print(f"警告: 在 {class_path} 中未找到图片")
            continue

        print(f"处理 '{class_name}' 类别，共 {len(image_paths)} 张图片")
        train_paths, temp_paths = train_test_split(image_paths, train_size=train_ratio, random_state=42)
        relative_ratio = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(temp_paths, train_size=relative_ratio, random_state=42)

        for paths, target_dir in [(train_paths, train_dir), (val_paths, val_dir), (test_paths, test_dir)]:
            for src_path in tqdm(paths, desc=f"复制到 {os.path.basename(target_dir)}/{class_name}"):
                dst_path = os.path.join(target_dir, class_name, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)

    print("\n数据集划分完成:")
    for dataset, directory in [("训练集", train_dir), ("验证集", val_dir), ("测试集", test_dir)]:
        total = sum(len(os.listdir(os.path.join(directory, class_name))) for class_name in class_names)
        print(f"  {dataset} 总计: {total} 张图片")

    return train_dir, val_dir, test_dir


if __name__ == '__main__':
    split_dataset(source_dir, output_dir, train_ratio, val_ratio, test_ratio)