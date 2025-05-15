from scipy.io import loadmat
import numpy as np
from PIL import Image
import tqdm
import logging
from pathlib import Path
import json
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 牛津花卉102数据集的花卉种类名称
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily"
]


def get_safe_folder_name(name):
    """将花卉名称转换为安全的文件夹名称"""
    # 将空格替换为下划线，移除特殊字符
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[\s-]+', '_', safe_name).lower()
    return safe_name


def load_labels(path):
    """加载标签文件并返回标签数组"""
    try:
        labels_data = loadmat(path)
        return np.array(labels_data['labels'][0]) - 1  # 减1使索引从0开始
    except Exception as e:
        logger.error(f"加载标签文件失败: {e}")
        raise


def get_image_paths(img_dir):
    """获取所有图像文件的路径并排序"""
    img_dir_path = Path(img_dir)
    if not img_dir_path.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    flower_paths = sorted([str(f) for f in img_dir_path.glob('*.jpg')])
    if not flower_paths:
        raise ValueError(f"在 {img_dir} 中没有找到图像文件")

    return flower_paths


def process_all_images(images, labels, output_folder, resize_dim=(256, 256)):
    """按花卉种类处理所有图像并保存到对应的类别目录"""
    output_path = Path(output_folder)

    # 确保输出文件夹存在
    if not output_path.exists():
        output_path.mkdir(parents=True)

    logger.info(f"正在将图像分类到 {output_folder}...")

    # 统计各类别图像数量
    class_counts = {}

    # 创建花卉ID到名称的映射字典
    flower_id_to_name = {}

    # 使用tqdm显示进度条
    for idx in tqdm.tqdm(range(len(images))):
        try:
            # 打开图像并调整大小
            img_path = images[idx]
            img = Image.open(img_path)
            img = img.resize(resize_dim, Image.Resampling.LANCZOS)

            # 获取标签和目标路径
            label = labels[idx]

            # 使用花卉的实际名称作为文件夹名
            if label < len(FLOWER_NAMES):
                flower_name = FLOWER_NAMES[label]
                safe_name = get_safe_folder_name(flower_name)

                # 记录ID到名称的映射关系
                flower_id_to_name[int(label)] = safe_name

                class_name = safe_name
            else:
                # 如果没有对应的名称，回退到使用标签号
                class_name = f"c{label}"

            class_path = output_path / class_name

            # 确保类别目录存在
            if not class_path.exists():
                class_path.mkdir(parents=True)

            # 保存图像
            dest_file = class_path / Path(img_path).name
            img.save(dest_file)

            # 更新类别计数
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        except Exception as e:
            logger.error(f"处理图像 {idx} 失败: {e}")

    # 保存ID到名称的映射关系
    mapping_file = output_path / "class_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(flower_id_to_name, f, indent=4)

    # 保存各类别的图像数量统计
    stats_file = output_path / "class_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(class_counts, f, indent=4)

    logger.info(f"已保存类别映射关系到 {mapping_file}")
    logger.info(f"已保存类别统计信息到 {stats_file}")
    logger.info(f"共处理了 {len(images)} 张图像，分为 {len(class_counts)} 个类别")


def main():
    """主函数"""
    try:
        # 定义文件路径
        base_dir = Path("../data/0xford_flowers102")
        image_labels_path = base_dir / "imagelabels.mat"
        images_dir = base_dir / "102flowers/jpg"
        results_dir = base_dir / "results"

        # 加载标签
        logger.info("加载标签文件...")
        labels = load_labels(image_labels_path)

        # 获取图像路径
        logger.info("获取图像路径...")
        image_paths = get_image_paths(images_dir)

        # 处理所有图像并按类别分类
        process_all_images(image_paths, labels, results_dir)

        logger.info("处理完成！所有图像已按花卉种类分类到 results 文件夹")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()