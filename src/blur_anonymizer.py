import cv2
import numpy as np
from PIL import Image

def load_blur_pipeline(device: str = "cpu"):
    """
    Returns None since OpenCV blurring runs purely on the CPU 
    and doesn't require loading a heavy neural network.
    """
    return None

def anonymize_blur(image: Image.Image, pipe=None, device: str = "cpu") -> Image.Image:
    """
    Applies Gaussian Blur to the head region of the image based on spatial heuristics.
    Market-1501 images are relatively consistent bounding boxes of 64x128.
    """
    # Convert PIL Image to OpenCV format (RGB -> BGR)
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    h, w = cv_img.shape[:2]
    
    # Calculate the region of interest (top ~25% for the head/face)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    y1, y2 = 0, int(h * 0.28)
    
    roi = cv_img[y1:y2, x1:x2]
    
    # Apply a heavy Gaussian blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 30)
    
    # Place the blurred patch back onto the image
    cv_img[y1:y2, x1:x2] = blurred_roi
    
    # Convert OpenCV format back to PIL Image (BGR -> RGB)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
