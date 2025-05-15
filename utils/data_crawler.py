import csv
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from serpapi import GoogleSearch
from tqdm import tqdm

# ==== 配置区 ====
# 1. API Key
load_dotenv()  # 会从 .env 加载环境变量
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')  # Google/Bing via SerpAPI

# 2. 数据目录
BASE_DIR = Path('data/flowers')
RAW_DIR = BASE_DIR / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 3. 标签映射表：类别ID -> (英文名称, 中文别名)
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

# 4. 日志配置
logging.basicConfig(
    filename='download.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
FAILED_LOG_FILE = BASE_DIR / 'failed_downloads.csv'
REPORT_FILE = BASE_DIR / 'download_report.csv'


# ==== 图片爬取函数 ====
def fetch_from_serpapi(query, label_id, limit=30, output_dir=RAW_DIR):
    cnt = 0
    params = {
        'engine': 'google',
        'q': f'{query} flower',
        'tbm': 'isch',
        'ijn': '0',
        'api_key': SERPAPI_API_KEY,
    }
    search = GoogleSearch(params)
    result = search.get_dict()

    print(f"[{label_id}] Search for '{query}':")
    print(json.dumps(result.get('search_information', {}), ensure_ascii=False, indent=2))
    images = result.get('images_results', [])[:limit]
    print(f"[{label_id}] 找到 {len(images)} 张图片元数据\n")

    with FAILED_LOG_FILE.open('a', newline='', encoding='utf-8') as failfile:
        fail_writer = csv.writer(failfile)
        for idx, img in enumerate(tqdm(images, desc=f"下载 类别 {label_id}", leave=False)):
            url = img.get('original') or img.get('link') or img.get('thumbnail')
            if not url:
                continue
            ext = Path(url).suffix.split('?')[0].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                ext = '.jpg'
            fname = f"{label_id}_{idx}{ext}"
            save_path = output_dir / fname

            for attempt in range(3):
                try:
                    resp = requests.get(
                        url,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                            'Referer': 'https://www.google.com'
                        },
                        timeout=5
                    )
                    resp.raise_for_status()
                    with open(save_path, 'wb') as f:
                        f.write(resp.content)
                    logging.info(f"成功下载: {save_path.name}")
                    cnt += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        logging.error(f"下载失败 [{fname}]: {e}")
                        fail_writer.writerow([label_id, fname, url, str(e)])
            time.sleep(0.1)
        return cnt

def main():
    # 初始化失败日志
    with open(FAILED_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['类别ID', '英文名', '图片链接', '错误信息'])

    report = []

    for label_id, (eng_name, zh_name) in LABEL_MAP.items():
        print(f"=== 正在爬取 类别 {label_id} ：{eng_name} ({zh_name}) ===")
        count = fetch_from_serpapi(query=eng_name, label_id=label_id, limit=30)
        report.append((label_id, eng_name, zh_name, count))

    print("所有类别爬取完成。")

    # 写入统计报告
    with open(REPORT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['类别ID', '英文名', '中文名', '成功下载数量'])
        writer.writerows(report)
    print(f"已生成统计报告：{REPORT_FILE}")


if __name__ == '__main__':
    main()

