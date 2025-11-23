Smart Watermark Remover 

这是一个基于 Python 的智能图片处理工具，专门用于自动检测并裁剪图片底部的水印、来源标识、广告横条或无关文字。

采用OCR 规则匹配 + LLM 智能仲裁的混合策略，既保证了常见水印（如微博、小红书）的处理速度，又能通过大模型精准识别不规则的复杂水印。

 功能特性

 图像增强预处理：内置 CLAHE 和 Gamma 校正，能够“照亮”隐蔽或半透明的浅色水印，提高识别率。
 混合检测机制:
      预筛选：利用 RapidOCR 识别文字，若命中关键词（如 `微博`, `@`, `uid` 等），直接通过几何计算进行物理裁剪，无需消耗 Token。
      LLM兜底：若 OCR 识别出未知文本，自动调用 LLM (豆包/Volcengine) 进行语义分析，判断是否为水印并计算裁剪比例。
      (Remark：支持中文路径读写，针对 CPU 环境优化（使用 ONNX Runtime），部署轻量。)

依赖环境:

基于 Python 3 ，主要依赖以下库：

*   opencv-python: 用于图像读取、预处理（灰度/对比度增强）及裁剪。
*   numpy: 用于图像矩阵运算。
*   requests: 用于调用 LLM API。
*   rapidocr_onnxruntime: 基于 ONNX 的轻量级 OCR 引擎，无需 GPU 即可快速运行。

 
