import cv2
import numpy as np
from PIL import Image

def load_silhouette_pipeline(device: str = "cpu"):
    """
    Returns None since OpenCV Canny Edge Detection runs purely on the CPU.
    """
    return None

def anonymize_silhouette(image: Image.Image, pipe=None, device: str = "cpu") -> Image.Image:
    """
    Applies Canny Edge Detection to strip all texturing/color and turn the 
    identity into a bare silhouette wireframe representing Level 4 anonymization.
    """
    # Convert PIL directly to Grayscale OpenCV numpy array
    cv_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply Canny Edge Detection
    # 100/200 are standard thresholds. You can tweak them based on dry run results.
    edges = cv2.Canny(cv_gray, 100, 200)
    
    # OSNet will expect 3-channel tensors later. Since Canny outputs 1-channel Grayscale,
    # we convert it back to a 3-channel (RGB) format where all 3 channels have the same edge data.
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)
