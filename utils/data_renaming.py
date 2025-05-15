import os
import re
from PIL import Image

# ==== 用户配置区域 ====
SRC_DIR = "./data/flowers/raw"         # 原始图像所在路径
DST_DIR = "../data/0xford_flowers102/results"   # 输出图像路径
os.makedirs(DST_DIR, exist_ok=True)

# ==== 类别映射表：类别编号 -> (英文名, 中文名) ====
LABEL_MAP = {
    "0": ("pink_primrose", "粉红报春花"),
    "1": ("hard_leaved_pocket_orchid", "硬叶袋兰"),
    "2": ("canterbury_bells", "坎特伯雷钟花"),
    "3": ("sweet_pea", "香豌豆"),
    "4": ("english_marigold", "英国万寿菊"),
    "5": ("tiger_lily", "虎百合"),
    "6": ("moon_orchid", "月兰"),
    "7": ("bird_of_paradise", "极乐鸟花"),
    "8": ("monkshood", "乌头"),
    "9": ("globe_thistle", "球蓟"),
    "10": ("snapdragon", "金鱼草"),
    "11": ("colts_foot", "款冬"),
    "12": ("king_protea", "国王珠宝玉兰"),
    "13": ("spear_thistle", "矛蓟"),
    "14": ("yellow_iris", "黄鸢尾"),
    "15": ("globe_flower", "地锦花"),
    "16": ("purple_coneflower", "紫锥花"),
    "17": ("peruvian_lily", "秘鲁百合"),
    "18": ("balloon_flower", "风铃草"),
    "19": ("giant_white_arum_lily", "巨型白天南星"),
    "20": ("fire_lily", "火百合"),
    "21": ("pincushion_flower", "缝球花"),
    "22": ("fritillary", "斗篷百合"),
    "23": ("red_ginger", "红姜花"),
    "24": ("grape_hyacinth", "葡萄风信子"),
    "25": ("corn_poppy", "玉蜀黍罂粟"),
    "26": ("prince_of_wales_feathers", "威尔士亲王羽毛"),
    "27": ("stemless_gentian", "无茎龙胆"),
    "28": ("artichoke", "朝鲜蓟"),
    "29": ("sweet_william", "石竹"),
    "30": ("carnation", "康乃馨"),
    "31": ("garden_phlox", "花园飞燕草"),
    "32": ("love_in_the_mist", "雾中之爱"),
    "33": ("mexican_aster", "墨西哥紫菀"),
    "34": ("alpine_sea_holly", "高山海刺球"),
    "35": ("ruby_lipped_cattleya", "红唇石斛兰"),
    "36": ("cape_flower", "开普花"),
    "37": ("great_masterwort", "舞鹤花"),
    "38": ("siam_tulip", "暹罗郁金香"),
    "39": ("lenten_rose", "忍冬玫瑰"),
    "40": ("barbeton_daisy", "巴贝顿雏菊"),
    "41": ("daffodil", "水仙"),
    "42": ("sword_lily", "剑兰"),
    "43": ("poinsettia", "一品红"),
    "44": ("bolero_deep_blue", "波莱罗深蓝"),
    "45": ("wallflower", "岩石花"),
    "46": ("marigold", "万寿菊"),
    "47": ("buttercup", "毛茛"),
    "48": ("oxeye_daisy", "菜豌豆菊"),
    "49": ("common_dandelion", "普通蒲公英"),
    "50": ("petunia", "矮牵牛"),
    "51": ("wild_pansy", "野雏菊"),
    "52": ("primula", "堇菜"),
    "53": ("sunflower", "向日葵"),
    "54": ("pelargonium", "老鹳草"),
    "55": ("bishop_of_llandaff", "兰德夫主教"),
    "56": ("gaura", "白云花"),
    "57": ("geranium", "天竺葵"),
    "58": ("orange_dahlia", "橙色大丽花"),
    "59": ("pink_yellow_dahlia", "粉黄大丽花"),
    "60": ("cautleya_spicata", "矛花姜"),
    "61": ("japanese_anemone", "日本银莲花"),
    "62": ("black_eyed_susan", "黑心金光菊"),
    "63": ("silverbush", "银灌木"),
    "64": ("californian_poppy", "加州罂粟"),
    "65": ("osteospermum", "银莲花"),
    "66": ("spring_crocus", "春番红花"),
    "67": ("bearded_iris", "硬毛鸢尾"),
    "68": ("windflower", "凤仙花"),
    "69": ("tree_poppy", "树罂粟"),
    "70": ("gazania", "矢车菊"),
    "71": ("azalea", "杜鹃花"),
    "72": ("water_lily", "睡莲"),
    "73": ("rose", "玫瑰"),
    "74": ("thorn_apple", "苦楝"),
    "75": ("morning_glory", "牵牛花"),
    "76": ("passion_flower", "西番莲"),
    "77": ("lotus", "莲花"),
    "78": ("toad_lily", "蟾蜍百合"),
    "79": ("anthurium", "凤尾蕉"),
    "80": ("frangipani", "鸡蛋花"),
    "81": ("clematis", "铁线莲"),
    "82": ("hibiscus", "木槿"),
    "83": ("columbine", "耧斗菜"),
    "84": ("desert_rose", "沙漠玫瑰"),
    "85": ("tree_mallow", "棉花藤"),
    "86": ("magnolia", "木兰"),
    "87": ("cyclamen", "仙客来"),
    "88": ("watercress", "西洋菜"),
    "89": ("canna_lily", "美人蕉"),
    "90": ("hippeastrum", "朱顶红"),
    "91": ("bee_balm", "荥草"),
    "92": ("ball_moss", "球藻"),
    "93": ("foxglove", "毛地黄"),
    "94": ("bougainvillea", "三角梅"),
    "95": ("camellia", "山茶花"),
    "96": ("mallow", "锦葵"),
    "97": ("mexican_petunia", "墨西哥牵牛"),
    "98": ("bromelia", "凤梨科植物"),
    "99": ("blanket_flower", "金光菊"),
    "100": ("trumpet_creeper", "茑藤"),
    "101": ("blackberry_lily", "黑莓百合")
}

# 每个类别一个计数器
label_counters = {v[0]: 0 for v in LABEL_MAP.values()}

# 成功/失败统计
success_count = 0
fail_count = 0

def sanitize_ext(filename):
    """清理非法扩展名，如 '.jpg!s2' → '.jpg'"""
    match = re.search(r'\.(jpg|jpeg|png|webp)', filename, re.IGNORECASE)
    return match.group(0).lower() if match else ".jpg"

def convert_and_save(src_path, dst_path):
    """转换图像格式并保存，同时打印处理状态"""
    global success_count, fail_count
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, format="JPEG", quality=90)
        print(f"✅ 图像处理成功：{src_path} -> {dst_path}")
        success_count += 1
    except Exception as e:
        print(f"⚠️ 图像处理失败：{src_path}，错误：{e}")
        fail_count += 1

# ==== 遍历所有原始文件 ====
for fname in os.listdir(SRC_DIR):
    if "_" not in fname:
        continue

    label_index = fname.split("_")[0]
    if label_index not in LABEL_MAP:
        continue

    class_name = LABEL_MAP[label_index][0]
    class_dir = os.path.join(DST_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    clean_ext = sanitize_ext(fname)
    src_path = os.path.join(SRC_DIR, fname)

    count = label_counters[class_name]
    new_fname = f"{class_name}_{count}.jpg"
    dst_path = os.path.join(class_dir, new_fname)

    convert_and_save(src_path, dst_path)
    label_counters[class_name] += 1

# ==== 输出总计信息 ====
print("\n📊 处理结果统计：")
print(f"✅ 成功图像数：{success_count}")
print(f"⚠️ 失败图像数：{fail_count}")
