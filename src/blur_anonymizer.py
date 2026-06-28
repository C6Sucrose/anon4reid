import os
import urllib.request
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


_detector = None
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"


def _get_model_path():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        print(f"Downloading face detection model to {model_path}...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete.")
    return model_path


def load_blur_pipeline(device: str = "cpu"):
    global _detector
    model_path = _get_model_path()
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.3,
    )
    _detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    return _detector


def anonymize_blur(image: Image.Image, pipe=None, device: str = "cpu") -> Image.Image:
    global _detector
    if _detector is None:
        load_blur_pipeline(device)

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    rgb_array = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
    results = _detector.detect(mp_image)

    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(w, bbox.origin_x + bbox.width)
            y2 = min(h, bbox.origin_y + bbox.height)

            if x2 > x1 and y2 > y1:
                roi = cv_img[y1:y2, x1:x2]
                cv_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 30)
    else:
        x1, x2 = int(w * 0.15), int(w * 0.85)
        y1, y2 = 0, int(h * 0.28)
        roi = cv_img[y1:y2, x1:x2]
        cv_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 30)

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
