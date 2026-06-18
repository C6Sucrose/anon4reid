"""
This module handles the image scaling logic required before and after running
any generative model in the ANON4REID pipeline. Since Market-1501 images are
natively 128x64 pixels, they are far too small for generative models like
Stable Diffusion Inpainting or Stable Diffusion RAD to process without producing
noise or garbage output. To solve this, every image is upscaled to 512x256 before
being passed to a model, and then downscaled back to 128x64 after the model outputs
its result. This module provides the core load, upscale, downscale, and save
functions that are shared across all anonymization methods in this project.
"""

import cv2
import numpy as np
from PIL import Image


ORIGINAL_SIZE = (64, 128)
MODEL_SIZE    = (256, 512)


def upscale(image: Image.Image) -> Image.Image:
    return image.resize(MODEL_SIZE, Image.LANCZOS)


def downscale(image: Image.Image) -> Image.Image:
    return image.resize(ORIGINAL_SIZE, Image.LANCZOS)


def preprocess(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image = upscale(image)
    return image


def postprocess(image: Image.Image) -> Image.Image:
    return downscale(image)


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def save_image(image: Image.Image, save_path: str):
    image.save(save_path)
    print(f"Saved → {save_path}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    # Grab a real image from your dataset
    test_image_path = "../anon4reid/data/raw/Market-1501/query/0001_c1s1_001051_00.jpg"

    if os.path.exists(test_image_path):
        img = load_image(test_image_path)
        print(f"Original size : {img.size}")   # (64, 128)

        upscaled = preprocess(img)
        print(f"Upscaled size : {upscaled.size}")  # (256, 512)

        downscaled = postprocess(upscaled)
        print(f"Downscaled size: {downscaled.size}")  # (64, 128)

        save_image(upscaled,   "../anon4reid/outputs/results/test_upscaled.jpg")
        save_image(downscaled, "../anon4reid/outputs/results/test_downscaled.jpg")
    else:
        print(f"Image not found at {test_image_path}, adjust path")