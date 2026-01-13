import os
import logging

# 禁用 Paddle 的联网模型检查 (加速启动)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
from module.logger import logger

# 防止双重打印
# 这行代码的意思是：logger 的消息到此为止，不要再传给 Root Logger 了
logger.propagate = False

# 手动设置日志级别来代替 show_log=False
logging.getLogger("ppocr").setLevel(logging.WARNING)


class OcrAgent:
    def __init__(self):
        logger.info("Initializing PaddleOCR...")

        # =====================================================
        # 【修改点】删掉了 show_log=False，防止报错
        # =====================================================
        self.ocr = PaddleOCR(use_angle_cls=False, lang="ch")

        logger.info("PaddleOCR Initialized.")

    def predict(self, image_crop):
        """
        识别传入图片(裁剪后的)中的文字
        """
        # 增加容错：如果是空图，直接返回空字符串
        if image_crop is None or image_crop.size == 0:
            return ""

        try:
            # result 结构可能因版本不同而微调，加个 try-except 更稳
            result = self.ocr.ocr(image_crop, cls=False)

            if not result or result[0] is None:
                return ""

            # PaddleOCR 返回的结构通常是 [[[[x,y]..], (text, conf)], ...]
            text_list = [line[1][0] for line in result[0]]
            return "".join(text_list)

        except Exception as e:
            logger.warning(f"OCR Error: {e}")
            return ""


# 单例模式
ocr_agent = OcrAgent()