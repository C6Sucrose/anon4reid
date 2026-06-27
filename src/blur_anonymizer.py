import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


_face_detection = None


def load_blur_pipeline(device: str = "cpu"):
    global _face_detection
    mp_face = mp.solutions.face_detection
    _face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    return _face_detection


def anonymize_blur(image: Image.Image, pipe=None, device: str = "cpu") -> Image.Image:
    global _face_detection
    if _face_detection is None:
        load_blur_pipeline(device)

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    results = _face_detection.process(rgb_img)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)

            if x2 > x1 and y2 > y1:
                roi = cv_img[y1:y2, x1:x2]
                cv_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 30)
    else:
        # Fallback: blur top 28% of image (spatial heuristic for 128x64 pedestrian crops)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        y1, y2 = 0, int(h * 0.28)
        roi = cv_img[y1:y2, x1:x2]
        cv_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 30)

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
