"""
This script serves as a dry run evaluation of all available anonymization pipelines
in the ANON4REID project. It randomly selects 10 images from the Market-1501 query
set and runs them through whichever anonymization pipelines are currently implemented
and available. The outputs are saved to the outputs/results folder organized by pipeline
name so the team can visually review and compare the results. Based on this review,
parameters such as blur radius, mask size, and diffusion prompt strength will be tuned
and locked in for the full dataset run. The script automatically detects which pipeline
modules exist and skips any that have not been implemented yet.
"""

import os
import random
import importlib
import torch
from PIL import Image


QUERY_DIR     = "data/raw/Market-1501/query"
OUTPUT_DIR    = "outputs/results/dry_run"
NUM_IMAGES    = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def get_random_images(query_dir: str, n: int = 10):
    all_images = [
        os.path.join(query_dir, f)
        for f in os.listdir(query_dir)
        if f.endswith(".jpg")
    ]
    return random.sample(all_images, min(n, len(all_images)))


def run_pipeline(name, load_fn, anonymize_fn, images, output_dir):
    print(f"\n{'='*50}")
    print(f"Running pipeline: {name}")
    print(f"{'='*50}")

    pipeline_output_dir = os.path.join(output_dir, name)
    os.makedirs(pipeline_output_dir, exist_ok=True)

    print(f"Loading {name} model...")
    pipe = load_fn(DEVICE)

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        fname = os.path.basename(img_path)

        print(f"  [{i+1}/{len(images)}] Processing {fname}...")
        result = anonymize_fn(img, pipe, device=DEVICE)

        save_path = os.path.join(pipeline_output_dir, fname)
        result.save(save_path)

    print(f"Done! Results saved to {pipeline_output_dir}")


def check_pipeline(module_name, load_fn_name, anonymize_fn_name):
    """
    Checks if a pipeline module exists and has the required functions.
    Returns (load_fn, anonymize_fn) if available, otherwise returns (None, None).
    """
    try:
        module = importlib.import_module(module_name)
        load_fn     = getattr(module, load_fn_name)
        anonymize_fn = getattr(module, anonymize_fn_name)
        print(f"  ✔ {module_name} found and loaded")
        return load_fn, anonymize_fn
    except (ModuleNotFoundError, AttributeError):
        print(f"  ✘ {module_name} not found, skipping")
        return None, None


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"\nSelecting {NUM_IMAGES} random images from query set...")
    images = get_random_images(QUERY_DIR, NUM_IMAGES)
    print(f"Selected {len(images)} images")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define all 4 pipelines — script will skip any that don't exist yet
    pipelines = [
        {
            "name":         "lama",
            "module":       "lama_inpaint",
            "load_fn":      "load_lama_pipeline",
            "anonymize_fn": "anonymize_lama",
        },
        {
            "name":         "rad",
            "module":       "rad_pipeline",
            "load_fn":      "load_rad_pipeline",
            "anonymize_fn": "anonymize_rad",
        },
        {
            "name":         "blur",
            "module":       "blur_anonymizer",
            "load_fn":      "load_blur_pipeline",
            "anonymize_fn": "anonymize_blur",
        },
        {
            "name":         "silhouette",
            "module":       "silhouette_anonymizer",
            "load_fn":      "load_silhouette_pipeline",
            "anonymize_fn": "anonymize_silhouette",
        },
    ]

    print("\nChecking available pipelines...")
    available = []
    for p in pipelines:
        load_fn, anonymize_fn = check_pipeline(
            p["module"], p["load_fn"], p["anonymize_fn"]
        )
        if load_fn and anonymize_fn:
            available.append({
                "name":         p["name"],
                "load_fn":      load_fn,
                "anonymize_fn": anonymize_fn,
            })

    print(f"\n{len(available)}/{len(pipelines)} pipelines available")

    for p in available:
        run_pipeline(
            name         = p["name"],
            load_fn      = p["load_fn"],
            anonymize_fn = p["anonymize_fn"],
            images       = images,
            output_dir   = OUTPUT_DIR,
        )

    print(f"\n{'='*50}")
    print(f"Dry run complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Pipelines run: {[p['name'] for p in available]}")
    print(f"{'='*50}")