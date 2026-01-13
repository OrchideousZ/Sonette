import cv2
from ultralytics import YOLO
from module.logger import logger
from PIL import Image
import numpy as np

class YoloAgent:
    def __init__(self, model_path='./models/best.pt'):
        logger.info(f'Loading YOLO model from {model_path}')
        self.model = YOLO(model_path)

    def predict(self, image, conf_thres=0.6):
        # 1. 确保 image 是 numpy 数组 (OpenCV 格式)
        if isinstance(image, np.ndarray):
            # 2. 核心步骤：BGR -> RGB -> PIL Image
            # 这样这就变成了一个标准的、绝对正确的 RGB 图片对象
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            # 如果万一传进来已经是 PIL 了，就直接用
            image_pil = image

        # 3. 推理 (传入 PIL 对象)
        # verbose=False 不打印日志
        results = self.model.predict(image_pil, conf=conf_thres, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            detections.append({
                'label': label,
                'conf': conf,
                'box': [x1, y1, x2, y2],
                'center': (center_x, center_y)
            })

        return detections


# 单例模式，全局只加载一次模型
yolo_agent = YoloAgent()