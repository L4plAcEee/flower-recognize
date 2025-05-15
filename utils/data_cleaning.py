import csv
from pathlib import Path
from PIL import Image
import imagehash
import cv2

# 感知哈希记录文件路径
HASHES_FILE = Path('data/flowers/hashes.csv')
# 原始图片目录路径
RAW_DIR = Path('data/flowers/raw')

def load_hashes():
    """
    加载已保存的图片哈希值，用于后续去重比对。
    如果哈希文件不存在，返回空字典。
    """
    if not HASHES_FILE.exists():
        return {}
    with open(HASHES_FILE, newline='') as csvf:
        reader = csv.reader(csvf)
        return {rows[0]: rows[1] for rows in reader}

def save_hash(hash_dict):
    """
    将当前哈希字典保存至本地 CSV 文件，供后续去重使用。
    """
    with open(HASHES_FILE, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        for fname, h in hash_dict.items():
            writer.writerow([fname, h])

def is_blurry(img_path, thresh=100.0):
    """
    判断图片是否模糊。
    参数：
        img_path: 图片路径
        thresh: 模糊判定的阈值（越低越容易被判为模糊）
    返回：
        若图像方差小于阈值，则认为是模糊图，返回 True；否则返回 False。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
    return fm < thresh

def dedupe_and_clean(output_dir=RAW_DIR):
    """
    对指定目录中的图片进行清洗和去重，包括以下操作：
    1. 删除无法打开的损坏图片；
    2. 删除分辨率小于 300x300 的图片；
    3. 删除模糊图片；
    4. 删除感知哈希重复的图片。
    清洗完成后更新哈希字典。
    """
    hashes = load_hashes()
    updated = False

    for img_path in output_dir.glob('*.*'):
        try:
            with Image.open(img_path) as img:
                # 分辨率过滤
                if img.width < 300 or img.height < 300:
                    img_path.unlink()
                    continue

                # 模糊过滤
                if is_blurry(img_path):
                    img_path.unlink()
                    continue

                # 感知哈希去重
                h = str(imagehash.phash(img))
        except Exception:
            img_path.unlink()
            continue

        if h in hashes.values():
            img_path.unlink()
        else:
            hashes[str(img_path.name)] = h
            updated = True

    # 如果哈希记录有更新，则保存到本地文件
    if updated:
        save_hash(hashes)

if __name__ == '__main__':
    dedupe_and_clean()