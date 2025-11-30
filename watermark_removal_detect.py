import os
import cv2
import numpy as np
import requests
import base64
import json
from pathlib import Path
import time
from collections import defaultdict

# ---------- 配置区域 -------------
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
API_KEY = "95764e6c-0710-4095-ab55-a98301cebbc9"
MODEL = "doubao-seed-1-6-251015"

# 文件夹配置
SOURCE_IMAGE_FOLDER = "/root/yxc/project/热门微博img copy"
CLEANED_IMAGE_FOLDER = "/root/yxc/project/热门微博img_cleaned"
OUTPUT_JSON_FOLDER = "/root/yxc/project/post_analysis_results"
FINAL_JSON_PATH = "/root/yxc/project/all_posts_analysis.json"

# API调用延迟
API_DELAY = 0.5

# ============ 去水印配置 ============
BRAND_KEYWORDS = [
    "huawei", "华为", "pura", "xmage", "mate", "p40", "p50", "p60", "p70", "p80",
    "nova", "leica", "徕卡", "oppo", "vivo", "reno", "find", "iqoo", "nex", 
    "zeiss", "蔡司", "xiaomi", "小米", "redmi", "红米", "poco", "civi",
    "realme", "oneplus", "一加", "honor", "荣耀", "meizu", "魅族",
    "samsung", "三星", "galaxy", "iphone", "apple", "苹果",
    "sony", "索尼", "xperia", "pixel", "nothing",
    "shot on", "拍摄于", "photographed", "captured", 
    "hasselblad", "哈苏", "ultra", "pro", "max",
    "50mm", "35mm", "85mm", "portrait", "night mode", "夜景",
    "f/1", "f/2", "f/3", "iso", "1/", "mm ",
]

BLACKLIST_KEYWORDS = [
    "微博", "weibo", "小红书", "red", "douyin", "抖音", "快手", 
    "@", "©", "uid", "摄影", "出品", "来源", "net", "com",
    "视频", "video", "bilibili", "知乎", "ins", "instagram",
    "twitter", "tiktok", "youtube", "水印", "版权"
]

# -------------------初始化 RapidOCR---------------
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR(det_use_gpu=False, cls_use_gpu=False, rec_use_gpu=False)
    print("✓ OCR引擎初始化成功")
except ImportError:
    print("✗ 缺少 rapidocr_onnxruntime 库，请运行: pip install rapidocr_onnxruntime")
    exit()

# ------------------ 工具函数 ------------------

def safe_imread(path):
    """安全读取图片"""
    try:
        if not os.path.exists(path): return None
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except: 
        return None

def safe_imwrite(path, img):
    """安全保存图片"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(str(path))[1] or ".jpg"
        cv2.imencode(ext, img)[1].tofile(str(path))
        return True
    except: 
        return False

def enhance_watermark(img_bgr):
    """图像增强"""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray) 
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    except:
        return img_bgr

def check_keywords(text_lower, keyword_list):
    """检查关键词"""
    for kw in keyword_list:
        if kw.lower() in text_lower:
            return kw
    return None

def check_edge_density(image_path, check_height_ratio=0.15, threshold=0.03):
    """使用 Canny 边缘检测快速判断底部是否有丰富纹理"""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            return True, 0.0
        
        h, w = img.shape
        check_h = int(h * check_height_ratio)
        bottom_area = img[h-check_h:h, 0:w]
        edges = cv2.Canny(bottom_area, 100, 200)
        edge_density = np.count_nonzero(edges) / (edges.size + 1e-5)
        has_text = edge_density >= threshold
        
        return has_text, edge_density
        
    except Exception as e:
        return True, 0.0

def analyze_watermark(image_path):
    """分析水印（检测四角+底部）"""
    try:
        img = safe_imread(image_path)
        if img is None: 
            return "ERROR", [], "读取失败", None
        
        h, w = img.shape[:2]
        corner_ratio = 0.25
        bottom_ratio = 0.4
        
        regions = {
            "top-left": img[0:int(h*corner_ratio), 0:int(w*corner_ratio)],
            "top-right": img[0:int(h*corner_ratio), int(w*(1-corner_ratio)):w],
            "bottom-left": img[int(h*(1-corner_ratio)):h, 0:int(w*corner_ratio)],
            "bottom-right": img[int(h*(1-corner_ratio)):h, int(w*(1-corner_ratio)):w],
            "bottom-center": img[int(h*(1-bottom_ratio)):h, :]
        }
        
        watermark_locations = []
        all_texts = []
        
        for location, region_img in regions.items():
            enhanced_img = enhance_watermark(region_img)
            result, _ = ocr_engine(enhanced_img)
            
            if not result:
                continue
            
            filtered_result = [line for line in result if line[2] > 0.3]
            
            if not filtered_result:
                continue
            
            region_text = " ".join([line[1] for line in filtered_result])
            has_watermark = False
            
            for line in filtered_result:
                text_lower = line[1].lower()
                if check_keywords(text_lower, BRAND_KEYWORDS):
                    has_watermark = True
                    break
            
            if not has_watermark:
                for line in filtered_result:
                    text_lower = line[1].lower()
                    if check_keywords(text_lower, BLACKLIST_KEYWORDS):
                        has_watermark = True
                        break
            
            if has_watermark and len(region_text.replace(" ", "")) >= 2:
                watermark_locations.append(location)
                all_texts.append(f"[{location}]: {region_text}")
        
        if watermark_locations:
            return "HAS_WATERMARK", watermark_locations, "检测到水印", " | ".join(all_texts)
        else:
            return "CLEAN", [], "无水印", None

    except Exception as e:
        return "ERROR", [], str(e), None

def crop_watermark_directly(image_path, detected_locations):
    """直接裁剪水印（不调用LLM）"""
    try:
        img = safe_imread(image_path)
        if img is None: 
            return False, None, "读取失败"
        
        h, w = img.shape[:2]
        has_top = any(loc in detected_locations for loc in ["top-left", "top-right"])
        has_bottom = any(loc in detected_locations for loc in ["bottom-left", "bottom-right", "bottom-center"])
        crop_ratio = 0.1
        
        if has_bottom and not has_top:
            new_h = int(h * (1 - crop_ratio))
            cropped_img = img[:new_h, :]
            msg = f"OCR检测到底部水印，直接裁剪{crop_ratio:.1%}"
            return True, cropped_img, msg
            
        elif has_top and not has_bottom:
            crop_pixels = int(h * crop_ratio)
            cropped_img = img[crop_pixels:h, :]
            msg = f"OCR检测到顶部水印，直接裁剪{crop_ratio:.1%}"
            return True, cropped_img, msg
            
        elif has_top and has_bottom:
            new_h = int(h * (1 - crop_ratio))
            cropped_img = img[:new_h, :]
            msg = f"OCR检测到上下水印，裁剪底部{crop_ratio:.1%}"
            return True, cropped_img, msg
            
        else:
            return False, img, "未知水印位置"
            
    except Exception as e:
        return False, None, f"裁剪失败: {e}"

def remove_watermark_llm(image_path, detected_locations):
    """调用LLM判断水印位置并裁剪"""
    try:
        img = safe_imread(image_path)
        if img is None: 
            return False, None, "读取失败"
        
        h, w = img.shape[:2]
        max_size = 1024
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
        else:
            img_resized = img.copy()
        
        _, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""仔细检查图片的四个角落和底部边缘是否有水印、品牌标识、拍摄参数等。

注意：OCR未检测到明显水印，但边缘密度检测显示可能有文字，请仔细确认。

返回严格JSON格式：
{{
    "has_watermark": true或false,
    "crop_direction": "top"或"bottom"或"none",
    "crop_ratio": 0.0到0.25之间的浮点数,
    "reason": "简短说明（中文）"
}}

裁剪方向说明：
- "top": 从顶部裁剪（用于顶部角落水印）
- "bottom": 从底部裁剪（用于底部角落或底部中心水印）
- "none": 无需裁剪

要求：
- crop_ratio是要裁掉的比例，最大不超过0.25（25%）
- 只有明确看到水印时才裁剪
- 如果水印很小，crop_ratio设为0.05-0.10即可
- 如果不确定是否是水印，设为none"""
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        payload = {
            "model": MODEL,
            "messages": [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            "temperature": 0.1
        }
        
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            clean = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            
            if result.get('has_watermark') and result.get('crop_direction') != 'none':
                crop_ratio = min(result.get('crop_ratio', 0), 0.25)
                direction = result.get('crop_direction', 'bottom')
                
                if crop_ratio > 0:
                    if direction == 'top':
                        crop_pixels = int(h * crop_ratio)
                        cropped_img = img[crop_pixels:h, :]
                        msg = f"从顶部裁剪{crop_ratio:.1%}: {result.get('reason', '')}"
                    else:
                        new_h = int(h * (1 - crop_ratio))
                        cropped_img = img[:new_h, :]
                        msg = f"从底部裁剪{crop_ratio:.1%}: {result.get('reason', '')}"
                    
                    return True, cropped_img, msg
            
            return False, img, f"LLM判定无需裁剪: {result.get('reason', '')}"
        else:
            return False, img, f"API错误: {resp.status_code}"
            
    except Exception as e:
        return False, None, f"LLM调用失败: {e}"

# ------------------ 关联性分析函数 ------------------

def analyze_post_relevance(image_paths):
    """分析一个post下所有图片的关联性（中英文双版本）"""
    if not image_paths:
        return None
    
    post_id = os.path.basename(image_paths[0]).split('_')[0]
    
    print(f"\n{'='*60}")
    print(f"分析 Post {post_id} 的图片关联性")
    print(f"图片数量: {len(image_paths)}")
    print(f"{'='*60}\n")
    
    # 准备所有图片的base64编码
    image_data_list = []
    for i, path in enumerate(image_paths, 1):
        img = safe_imread(path)
        if img is None:
            print(f"    图片 {i} 读取失败: {path}")
            continue
        
        max_size = 800
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        image_data_list.append({
            "image_id": os.path.basename(path).split('_')[1].split('.')[0],
            "filename": os.path.basename(path),
            "base64": base64_img
        })
    
    if not image_data_list:
        return None
    
    # Step 1: 调用LLM识别整体主题
    print("   Step 1: 识别Post主题...")
    time.sleep(API_DELAY)
    
    topic_prompt = f"""这是一个社交媒体帖子的{len(image_data_list)}张图片。请分析这个帖子的主题。

要求：
1. 用一句话概括这个帖子的主题（中英文）
2. 解释你的推理过程（中英文）

返回严格的JSON格式：
{{
    "topic_zh": "主题（中文，15字以内）",
    "topic_en": "Topic (English, within 10 words)",
    "reasoning_zh": "推理过程（中文，50字以内）",
    "reasoning_en": "Reasoning process (English, within 40 words)"
}}"""
    
    content_parts = [{"type": "text", "text": topic_prompt}]
    for img_data in image_data_list:
        content_parts.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}
        })
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0.3
    }
    
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code != 200:
            print(f"  API错误: {resp.status_code}")
            return None
        
        content = resp.json()['choices'][0]['message']['content']
        clean = content.replace("```json", "").replace("```", "").strip()
        topic_result = json.loads(clean)
        
        print(f"  ✓ 主题识别完成:")
        print(f"    中文: {topic_result['topic_zh']}")
        print(f"    英文: {topic_result['topic_en']}")
        
    except Exception as e:
        print(f"   主题识别失败: {e}")
        return None
    
    # Step 2: 逐张分析图片关联性
    print(f"\n   Step 2: 分析每张图片的关联性...")
    
    image_results = []
    for i, img_data in enumerate(image_data_list, 1):
        print(f"    [{i}/{len(image_data_list)}] 分析 {img_data['filename']}...", end=" ")
        
        time.sleep(API_DELAY)
        
        relevance_prompt = f"""主题（中文）：{topic_result['topic_zh']}
主题（英文）：{topic_result['topic_en']}

判断这张图片是否与上述主题强相关。

返回严格JSON格式：
{{
    "is_related": true或false,
    "reasoning_zh": "判断理由（中文，30字以内）",
    "reasoning_en": "Reasoning (English, within 25 words)"
}}

强相关的标准：
- 图片直接展示主题内容
- 图片是主题的核心组成部分
- 图片与主题有明确的因果或从属关系

弱相关（判定为false）：
- 仅仅是背景、装饰、无关截图
- 与主题无直接联系的随机图片"""
        
        payload = {
            "model": MODEL,
            "messages": [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": relevance_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}}
                ]
            }],
            "temperature": 0.2
        }
        
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f" API错误")
                continue
            
            content = resp.json()['choices'][0]['message']['content']
            clean = content.replace("```json", "").replace("```", "").strip()
            rel_result = json.loads(clean)
            
            image_results.append({
                "image_id": img_data['image_id'],
                "filename": img_data['filename'],
                "is_related": rel_result['is_related'],
                "reasoning_zh": rel_result['reasoning_zh'],
                "reasoning_en": rel_result['reasoning_en']
            })
            
            status = "✓ 强相关" if rel_result['is_related'] else "○ 弱相关"
            print(f"{status}")
            
        except Exception as e:
            print(f" 失败: {e}")
            continue
    
    # 汇总结果
    result = {
        "post_id": post_id,
        "topic_zh": topic_result['topic_zh'],
        "topic_en": topic_result['topic_en'],
        "topic_reasoning_zh": topic_result['reasoning_zh'],
        "topic_reasoning_en": topic_result['reasoning_en'],
        "images": image_results
    }
    
    return result

# ------------------ 去水印处理流程 ------------------

def process_single_image_watermark(source_path, cleaned_path):
    """
    处理单张图片的去水印
    返回: (是否成功, 消息)
    """
    # 步骤1: OCR分析水印
    status, locations, msg, ocr_text = analyze_watermark(source_path)
    
    if status == "ERROR":
        # 读取失败，直接复制原图
        img = safe_imread(source_path)
        if img is not None:
            safe_imwrite(cleaned_path, img)
            return True, f"读取错误，保留原图"
        return False, f"完全失败"
    
    # 步骤2: 根据OCR结果决定后续流程
    if status == "HAS_WATERMARK":
        # OCR检测到水印 → 直接裁剪
        cropped, result_img, crop_msg = crop_watermark_directly(source_path, locations)
        
        if cropped and result_img is not None:
            if safe_imwrite(cleaned_path, result_img):
                return True, crop_msg
            else:
                return False, "保存失败"
        else:
            # 裁剪失败，保留原图
            img = safe_imread(source_path)
            if img is not None:
                safe_imwrite(cleaned_path, img)
            return True, f"裁剪失败，保留原图"
    
    else:  # status == "CLEAN"
        # OCR认为安全 → 进行边缘密度检查
        has_text, density = check_edge_density(source_path)
        
        if not has_text:
            # 边缘密度也安全 → 真正安全，直接复制
            img = safe_imread(source_path)
            if img is not None:
                safe_imwrite(cleaned_path, img)
                return True, "双重确认安全"
            return False, "读取失败"
        else:
            # 边缘密度异常 → 调用LLM二次确认
            time.sleep(API_DELAY)
            cropped, result_img, llm_msg = remove_watermark_llm(source_path, [])
            
            if cropped and result_img is not None:
                if safe_imwrite(cleaned_path, result_img):
                    return True, llm_msg
                else:
                    return False, "保存失败"
            else:
                # LLM判定无需裁剪，保留原图
                img = safe_imread(source_path)
                if img is not None:
                    safe_imwrite(cleaned_path, img)
                return True, llm_msg

# ------------------ 批量处理主函数 ------------------

def batch_process_all_posts():
    """批量处理所有posts"""
    print(f"\n{'#'*70}")
    print(f"# 批量处理模式")
    print(f"# 源文件夹: {SOURCE_IMAGE_FOLDER}")
    print(f"# 清洁图片文件夹: {CLEANED_IMAGE_FOLDER}")
    print(f"# JSON输出文件夹: {OUTPUT_JSON_FOLDER}")
    print(f"# 最终汇总JSON: {FINAL_JSON_PATH}")
    print(f"{'#'*70}\n")
    
    # 创建输出文件夹
    os.makedirs(CLEANED_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)
    
    # 1. 收集所有图片并按post分组
    post_images = defaultdict(list)
    
    for filename in os.listdir(SOURCE_IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                post_id = filename.split('_')[0]
                post_images[post_id].append(filename)
            except:
                continue
    
    # 按post_id排序
    sorted_posts = sorted(post_images.keys(), key=lambda x: int(x))
    
    print(f"找到 {len(sorted_posts)} 个Posts，共 {sum(len(v) for v in post_images.values())} 张图片\n")
    
    all_results = {}
    
    # 2. 处理每个post
    for idx, post_id in enumerate(sorted_posts, 1):
        print(f"\n{'='*70}")
        print(f"处理 Post {post_id} ({idx}/{len(sorted_posts)})")
        print(f"{'='*70}")
        
        image_files = sorted(post_images[post_id], 
                           key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"\n阶段1: 去水印处理 ({len(image_files)} 张图片)")
        print(f"{'-'*70}")
        
        cleaned_paths = []
        
        for i, filename in enumerate(image_files, 1):
            source_path = os.path.join(SOURCE_IMAGE_FOLDER, filename)
            cleaned_path = os.path.join(CLEANED_IMAGE_FOLDER, filename)
            
            print(f"  [{i}/{len(image_files)}] {filename}... ", end="")
            
            success, msg = process_single_image_watermark(source_path, cleaned_path)
            
            if success:
                print(f"✓ {msg}")
                cleaned_paths.append(cleaned_path)
            else:
                print(f"✗ {msg}")
        
        # 3. 关联性分析
        if cleaned_paths:
            print(f"\n阶段2: 图片关联性分析")
            print(f"{'-'*70}")
            
            relevance_result = analyze_post_relevance(cleaned_paths)
            
            if relevance_result:
                # 保存单个post的JSON
                post_json_path = os.path.join(OUTPUT_JSON_FOLDER, f"post_{post_id}_analysis.json")
                with open(post_json_path, 'w', encoding='utf-8') as f:
                    json.dump(relevance_result, f, ensure_ascii=False, indent=2)
                
                all_results[post_id] = relevance_result
                
                print(f"\n✓ Post {post_id} 处理完成")
                print(f"  主题: {relevance_result['topic_zh']}")
                related_count = sum(1 for img in relevance_result['images'] if img['is_related'])
                print(f"  强相关图片: {related_count}/{len(relevance_result['images'])}")
            else:
                print(f"\n✗ Post {post_id} 关联性分析失败")
        else:
            print(f"\n✗ Post {post_id} 没有成功处理的图片")
    
    # 4. 保存最终汇总JSON
    print(f"\n{'='*70}")
    print(f"保存最终汇总结果...")
    print(f"{'='*70}")
    
    with open(FINAL_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 所有处理完成！")
    print(f"  处理的Posts数量: {len(all_results)}/{len(sorted_posts)}")
    print(f"  清洁图片保存在: {CLEANED_IMAGE_FOLDER}")
    print(f"  单Post JSON保存在: {OUTPUT_JSON_FOLDER}")
    print(f"  最终汇总JSON: {FINAL_JSON_PATH}")

# ------------------ 入口 ------------------

if __name__ == "__main__":
    batch_process_all_posts()