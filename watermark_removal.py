import os
import cv2
import numpy as np
import requests
import base64
import json
from pathlib import Path

# ---------- 配置区域 -------------
# 这一块填 API Key
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
API_KEY = "95764e6c-0710-4095-ab55-a98301cebbc9"
MODEL = "doubao-seed-1-6-251015"

# 要处理的图片路径
IMAGE_PATH = "/root/yxc/output/热门微博_1779_图片_10_007X9jWzly1i5u8f3qbk3j31400u0q5e.jpg"

# -------------------初始化 RapidOCR---------------
try:
    from rapidocr_onnxruntime import RapidOCR
    # 启动 OCR 引擎。det=检测, cls=方向分类, rec=文字识别
    # 咱们这儿主要跑在 CPU 上，虽然没 GPU 快，但胜在稳定，哪里都能跑
    ocr_engine = RapidOCR(det_use_gpu=False, cls_use_gpu=False, rec_use_gpu=False) #初始化OCR
except ImportError:
    print(" 哎呀，缺少 rapidocr_onnxruntime 库，赶紧 pip install 一下！")
    exit()

# ------------------ 工具函数区域---------------

def enhance_watermark(img_bgr):
    """
    图像增强大法
    很多水印做得很鸡贼，颜色很淡。
    这个函数通过 CLAHE（自适应直方图均衡）和 Gamma 校正，
    强行把那些隐隐约约的水印给揪出来，让 OCR 能看清。
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) #将BGR 格式的彩色图像转换为灰度图像
        
        # 1. 对比度调整
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))# tileGridSize切割整张图片的参数，此时为64块 ;clipLimit=3.0 每一块的对比度阈值 超过阈值的部分被均匀化 相对来讲原本模糊的部分就会变得更清晰(水印)
        enhanced = clahe.apply(gray) 
        
        # 2. Gamma 校正：把暗部细节提亮，专门对付半透明水印
        gamma = 1.5 #控制亮度变换曲线的形状 ＞1代表提亮暗部(水印)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")# 生成 Gamma 校正查找表
        enhanced = cv2.LUT(enhanced, table) #cv2.LUT：OpenCV 的查找表变换函数，用预生成的table对图像enhanced进行逐像素映射，快速完成 Gamma 校正
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)#将增强后的灰度图像转换回BGR 彩色图像格式
    except Exception:
        # 如果增强过程崩了，就原样返回，别卡死
        return img_bgr

def safe_imread(path):
    """安全读取图片，防止中文路径报错"""
    try:
        if not os.path.exists(path): return None
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception: return None

def safe_imwrite(path, img):
    """安全保存图片，同样是为了兼容中文路径"""
    try:
        ext = os.path.splitext(str(path))[1] or ".jpg"
        cv2.imencode(ext, img)[1].tofile(str(path))
    except Exception: pass

def calculate_direct_crop_ratio(ocr_results, img_height, crop_offset_h):#根据 OCR 识别结果匹配预设关键词，计算图片底部需要裁剪的比例
    """
    精准切除
    如果发现了关键词，不需要 LLM 瞎猜，我们直接根据文字的位置来切。
    
    参数:
    - ocr_results: OCR 扫出来的结果列表
    - img_height: 原始图片的总高度
    - crop_offset_h: 底部裁剪区域的起始 Y 坐标（因为 OCR 是在底部切片上跑的）
    
    返回:
    - crop_ratio: 应该切掉底部百分之多少 (0.0 - 1.0)
    - hit_keyword: 命中了哪个关键词
    """
    # 这一串是“黑名单”，只要沾边，直接切！
    keywords = [
        "微博", "weibo", "小红书", "red", "douyin", "抖音", "快手", 
        "@", "©", "uid", "id", "号", "摄影", "出品", "来源", "net", "com",
        "视频", "video", "bilibili", "知乎"
    ]
    
    min_y_in_crop = float('inf') # 初始化读取坐标 原点在左上角 越大越靠下 可覆盖全张图片
    hit_kw = None # 初始化被命中的黑名单 默认为None
    
    found_keyword = False # 标记是否命中关键词(one-hot)
    
    if not ocr_results:
        return 0.0, None

    for line in ocr_results:# 每一行检测
        box, text, score = line[0], line[1], line[2]
        text_lower = text.lower() #统一大小写为小写 中文不变
        
        # 检查这句话里有没有关键词
        for kw in keywords:
            if kw in text_lower:
                found_keyword = True
                hit_kw = kw
                
                # box 是四个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # 我们要找这四个点里 Y 值最小的（也就是文字的最上沿）
                box_y_coords = [point[1] for point in box]
                current_min_y = min(box_y_coords)
                
                # 我们取所有命中关键词里，位置最高的那个（切得最狠，保证切干净）
                if current_min_y < min_y_in_crop:
                    min_y_in_crop = current_min_y
                
                # 找到了就不用继续遍历当前行的其他关键词了
                break
    
    if found_keyword:
        # 稍微往上多切一点点（比如 10 像素），留点余地，防止切到字头
        padding = 10 
        # 换算回原图的坐标
        # 裁剪点在原图的 Y = (底部区域起始Y) + (文字在底部区域的Y) - (安全边距)
        cut_y_absolute = crop_offset_h + min_y_in_crop - padding
        
        # 防止切过头（变成负数）
        if cut_y_absolute < 0: cut_y_absolute = 0
        
        # 计算切除比例： (总高度 - 保留高度) / 总高度
        # 其实就是：从 cut_y_absolute 往下都不要了
        ratio = (img_height - cut_y_absolute) / img_height
        return ratio, hit_kw
        
    return 0.0, None

def analyze_bottom_area(image_path, check_height_ratio=0.3):
    """
    【核心分析逻辑】
    1. 切出底部 30%
    2. 增强图片
    3. OCR 识别
    4. 决策：直接切？问 LLM？还是不管？
    """
    try:
        img = safe_imread(image_path)
        if img is None: return "ERROR", 0, "图片读取挂了"
        
        h, w = img.shape[:2] #返回图片的高和宽
        crop_h = int(h * check_height_ratio) # 底部区域的高度
        start_y = h - crop_h                 # 底部区域在原图的起始 Y
        bottom_img = img[start_y:h, :]    #切割的底部图片
        
        # 让隐形水印现形
        enhanced_img = enhance_watermark(bottom_img)
        
        # 开始识别
        result, _ = ocr_engine(enhanced_img) #调用OCR引擎

        # 如果啥字都没识出来，直接收工
        if not result:
            return "CLEAN", 0, "没发现文字，干净！"
            
        # 打印一下识别到的东西，方便调试
        print("OCR 扫描结果:")
        for line in result:
            if line[2] > 0.4: # 过滤掉置信度太低的垃圾数据
                print(f"      -> '{line[1]}'")

        # === 先查黑名单，中了直接切 ===
        direct_ratio, keyword = calculate_direct_crop_ratio(result, h, start_y)
        
        if direct_ratio > 0:
            # 限制一下最大裁剪比例，别把半张图都切了（上限设为 35%）
            final_ratio = min(direct_ratio, 0.35)
            return "DIRECT_CROP", final_ratio, f"命中关键词 [{keyword}]，执行物理超度"
            
        # === 如果有字，但不是关键词（比如路牌、衣服上的字） ===
        # 这时候拿不准，才去调用LLM
        return "ASK_LLM", 0, "发现未知文本，请求 LLM 仲裁"

    except Exception as e:
        print(f"预处理炸了: {e}")
        return "ERROR", 0, str(e)

def process_image_with_llm(image_path):
    """
    LLM 
    只有 OCR 拿不准的时候才调这个。
    """
    try:
        img = safe_imread(image_path)
        if img is None: return None
        # 压缩一下图片再发，省流量也跑得快
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = """
        检查图片【最底部】。判断是否有"水印"、"来源标识"、"广告"或"无关横条"。
        返回JSON: {"is_watermark": bool, "crop_ratio": float, "reason": string}
        crop_ratio范围0.0-0.3。
        """
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            "temperature": 0.1
        }
        
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            # 清理一下 LLM 可能返回的 markdown 格式符号
            clean = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
    except Exception as e:
        print(f"LLM 调用失败: {e}")
    return None

def crop_image(image_path, crop_ratio):
    """
    根据计算出的比例，把图片底部切掉。
    """
    try:
        img = safe_imread(image_path)
        if img is None: return
        h = img.shape[0]
        new_h = int(h * (1 - crop_ratio))
        
        p = Path(image_path)
        # 生成新文件名，加个 _cut 后缀
        out_path = p.parent / f"{p.stem}_cut{p.suffix}"
        
        safe_imwrite(out_path, img[:new_h])
        print(f"裁剪完成! 保存为: {out_path.name} (切除比例: {crop_ratio:.2%})")
    except Exception: pass

# ----------------- 主程序入口-----------------------
if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        print(f"开始处理: {os.path.basename(IMAGE_PATH)}")
        
        # 1. 先用 OCR 扫一遍，根据情况返回状态
        status, ratio, msg = analyze_bottom_area(IMAGE_PATH)
        
        print(f"预检状态: [{status}] -> {msg}")
        
        # 2. 根据状态分流处理
        if status == "DIRECT_CROP":
            # 命中关键词，直接切，省一次 API 调用
            print(" 触发快速裁剪模式 (无需 LLM)")
            crop_image(IMAGE_PATH, ratio)
            
        elif status == "ASK_LLM":
            # 有字但不知道是啥，问问 AI
            print("文字内容不确定，正在咨询 LLM...")
            res = process_image_with_llm(IMAGE_PATH)
            
            if res and res.get('is_watermark') and res.get('crop_ratio', 0) > 0:
                print(f"LLM 判定: 是水印 ({res['reason']})")
                crop_image(IMAGE_PATH, res['crop_ratio'])
            else:
                print("LLM 判定: 不是水印，无需裁剪")
                
        elif status == "CLEAN":
            print("图片底部很干净，无需操作")
            
        else:
            print("发生错误，跳过")
            
    else:
        print("找不到文件，请检查路径！")
