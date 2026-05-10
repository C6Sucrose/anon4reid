"""
This module implements the GAN-based anonymization method using LaMa (Large Mask inpainting),
a state-of-the-art inpainting model that uses Fourier convolutions to fill masked regions
with realistic background content. In the context of ANON4REID, LaMa is used to remove
the person from the image entirely and replace them with a plausible background, effectively
anonymizing their identity while preserving the scene context. The pipeline follows three
steps: upscale the input image to 512x256 so the model has enough resolution to work with,
run the inpainting model on the masked person region, then downscale the result back to
the original 128x64 Market-1501 size.
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from anonymizer import preprocess, postprocess


def load_lama_pipeline(device: str = "cuda"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    print(f"LaMa pipeline loaded on {device}.")
    return pipe


def generate_person_mask(image: Image.Image) -> Image.Image:
    """
    Generates a binary mask covering the central person region of the image.
    Since Market-1501 images are tightly cropped around the person, masking
    the central 60% width and 80% height reliably covers the subject without
    needing a separate segmentation model. The mask is white (255) where the
    person is and black (0) everywhere else, which is the format the inpainting
    model expects.
    """
    w, h = image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    y1, y2 = int(h * 0.1), int(h * 0.9)
    mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask).convert("RGB")


def anonymize_lama(
    image: Image.Image,
    pipe,
    prompt: str = "empty background, no person, floor, wall",
    negative_prompt: str = "person, human, body, blurry, distorted",
    device: str = "cuda"
) -> Image.Image:
    """
    Takes a single PIL Image, runs the full LaMa inpainting anonymization pipeline,
    and returns the anonymized image at the original 128x64 resolution. The function
    handles the upscaling, mask generation, inpainting, and downscaling internally
    so that callers only need to pass in a raw Market-1501 image and get back a
    transformed one of the same size.
    """
    image_up = preprocess(image)
    mask = generate_person_mask(image_up)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_up,
        mask_image=mask,
        height=512,
        width=256,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    return postprocess(result)


if __name__ == "__main__":
    test_image_path = "../anon4reid/data/raw/Market-1501/query/0001_c1s1_001051_00.jpg"

    if not os.path.exists(test_image_path):
        print("Image not found, adjust path")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    img  = Image.open(test_image_path).convert("RGB")
    pipe = load_lama_pipeline(device)

    result = anonymize_lama(img, pipe, device=device)

    os.makedirs("../anon4reid/outputs/results", exist_ok=True)
    result.save("../anon4reid/outputs/results/lama_result.jpg")
    print("Done! Saved to outputs/results/lama_result.jpg")