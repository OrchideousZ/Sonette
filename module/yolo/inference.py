import cv2
from ultralytics import YOLO
import numpy as np
from module.logger import logger


class YoloAgent:
    def __init__(self, model_path='./models/best.pt'):
        logger.info(f'Loading YOLO model from {model_path}')
        self.model = YOLO(model_path)

    def predict(self, image, conf_thres=0.6):
        # ============================================
        # 【核心修复】BGR 转 RGB
        # ALAS 的 image 是 BGR 格式，必须转成 RGB 喂给 YOLO
        # ============================================
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用转换后的 image_rgb 进行推理
        results = self.model.predict(image_rgb, conf=conf_thres, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 计算中心点 (ADB点击用)
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