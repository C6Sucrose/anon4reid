"""
This script runs the Stable Diffusion Inpainting pipeline on the Market-1501 dataset
to generate a fully anonymized version called Market-1501-SDInpaint. It processes all
three splits of the dataset: query, bounding_box_test, and bounding_box_train.
Each image is passed through the SD Inpainting pipeline which uses a geometric mask
covering the central torso region to remove identifying features and replaces them
with diffusion-generated content conditioned on a text prompt. The output directory
structure mirrors the original Market-1501 structure so it can be used as a drop-in
replacement for evaluation. Progress is tracked using tqdm and results are saved
iteratively after each image so that if the script is interrupted or disconnected it
can resume from where it left off without reprocessing already completed images.
"""

import os
import torch
from PIL import Image
from tqdm import tqdm

from sd_inpaint import load_sd_inpaint_pipeline, anonymize_sd_inpaint

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "raw", "Market-1501")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "Market-1501-SDInpaint")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS     = ["query", "bounding_box_test", "bounding_box_train"]


def get_images(split_dir: str):
    return [
        f for f in os.listdir(split_dir)
        if f.endswith(".jpg")
    ]


def already_processed(output_path: str) -> bool:
    """
    Checks if an image has already been processed by verifying if the output
    file already exists on disk. This allows the script to resume from where
    it left off if interrupted, without reprocessing already completed images.
    """
    return os.path.exists(output_path)


def process_split(split: str, pipe):
    input_split_dir  = os.path.join(INPUT_DIR, split)
    output_split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_split_dir, exist_ok=True)

    images = get_images(input_split_dir)
    print(f"\nProcessing {split}: {len(images)} images")

    skipped   = 0
    processed = 0

    for fname in tqdm(images, desc=split):
        input_path  = os.path.join(input_split_dir, fname)
        output_path = os.path.join(output_split_dir, fname)

        if already_processed(output_path):
            skipped += 1
            continue

        try:
            img    = Image.open(input_path).convert("RGB")
            result = anonymize_sd_inpaint(img, pipe, device=DEVICE)
            result.save(output_path)
            processed += 1
        except Exception as e:
            print(f"\nError processing {fname}: {e}, skipping...")
            continue

    print(f"  Done — {processed} processed, {skipped} skipped (already existed)")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Input  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Splits : {SPLITS}")

    print("\nLoading SD Inpaint pipeline...")
    pipe = load_sd_inpaint_pipeline(DEVICE)

    total_images = sum(
        len(get_images(os.path.join(INPUT_DIR, s)))
        for s in SPLITS
        if os.path.exists(os.path.join(INPUT_DIR, s))
    )
    print(f"\nTotal images to process: {total_images}")

    for split in SPLITS:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_dir):
            print(f"\nSkipping {split} — directory not found")
            continue
        process_split(split, pipe)

    print(f"\n{'='*50}")
    print(f"SD Inpaint dataset generation complete!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")