"""
This module implements the RAD (Realistic Anonymization with Diffusion) method using
Stable Diffusion v1.5 conditioned on ControlNet OpenPose and accelerated with LCM-LoRA.
Unlike the SD Inpainting approach which removes the person entirely via inpainting,
RAD replaces the person with a synthetically generated individual of different appearance
while preserving the original pose. This is achieved by first extracting the skeleton
keypoints from the input image using OpenPose, then using those keypoints as a conditioning
signal for Stable Diffusion to generate a new person in the same pose but with a completely
different identity. LCM-LoRA is applied to drastically reduce the number of inference steps
needed from 50 down to 4-8, making the pipeline practical for large scale anonymization of
the Market-1501 dataset. The image is upscaled to 512x256 before processing and downscaled
back to 128x64 after, following the same scaling logic as the SD Inpainting pipeline.
"""

import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LCMScheduler
)
from controlnet_aux import OpenposeDetector

from anonymizer import preprocess, postprocess


def load_rad_pipeline(device: str = "cuda"):
    print("Loading ControlNet OpenPose...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)

    print("Loading LCM-LoRA...")
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    print(f"RAD pipeline loaded on {device}.")
    return pipe


def extract_pose(image: Image.Image) -> Image.Image:
    """
    Extracts the OpenPose skeleton from the input image. The pose skeleton captures
    the person's body position and joint locations without retaining any identity
    information, making it a privacy-preserving conditioning signal for the diffusion
    model. The extracted pose is then used to guide Stable Diffusion into generating
    a new person in the exact same position and stance as the original subject.
    """
    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return detector(image)


def anonymize_rad(
    image: Image.Image,
    pipe,
    prompt: str = "a person, different appearance, full body, high quality, realistic",
    negative_prompt: str = "blurry, low quality, distorted, noise, same person",
    device: str = "cuda"
) -> Image.Image:
    """
    Takes a single PIL Image and returns an anonymized version where the original
    person has been replaced with a synthetically generated individual in the same
    pose. The function internally handles upscaling, pose extraction, diffusion
    inference, and downscaling so callers only need to pass in a raw Market-1501
    image and receive back a transformed one at the same 128x64 resolution.
    """
    image_up = preprocess(image)
    pose = extract_pose(image_up)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pose,
        height=512,
        width=256,
        num_inference_steps=8,
        guidance_scale=1.5,
        controlnet_conditioning_scale=0.8,
    ).images[0]

    return postprocess(result)


if __name__ == "__main__":
    test_image_path = "data/raw/Market-1501/query/0001_c1s1_001051_00.jpg"

    if not os.path.exists(test_image_path):
        print("Image not found, adjust path")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    img  = Image.open(test_image_path).convert("RGB")
    pipe = load_rad_pipeline(device)

    result = anonymize_rad(img, pipe, device=device)

    os.makedirs("outputs/results", exist_ok=True)
    result.save("outputs/results/rad_result.jpg")
    print("Done! Saved to outputs/results/rad_result.jpg")