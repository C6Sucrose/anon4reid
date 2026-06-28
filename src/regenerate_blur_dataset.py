"""
Regenerate Market-1501-Blur dataset using the updated blur_anonymizer
(MediaPipe face detection with spatial fallback).

Reads from:  data/raw/Market-1501/{query, bounding_box_test, bounding_box_train}
Writes to:   data/raw/Market-1501-Blur/{query, bounding_box_test, bounding_box_train}
"""

import os
import time
from PIL import Image
from tqdm import tqdm
from blur_anonymizer import load_blur_pipeline, anonymize_blur


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_root = os.path.join(base_dir, "data", "raw", "Market-1501")
    dst_root = os.path.join(base_dir, "data", "processed", "Market-1501-Blur")

    splits = ["query", "bounding_box_test", "bounding_box_train"]

    load_blur_pipeline()
    print("MediaPipe face detection loaded.\n")

    total_processed = 0
    t0 = time.time()

    for split in splits:
        src_dir = os.path.join(src_root, split)
        dst_dir = os.path.join(dst_root, split)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"Skipping {split} (not found at {src_dir})")
            continue

        fnames = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
        print(f"{split}: {len(fnames)} images")

        for fname in tqdm(fnames, desc=split):
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            result = anonymize_blur(img)
            result.save(os.path.join(dst_dir, fname))
            total_processed += 1

    elapsed = time.time() - t0
    print(f"\nDone. {total_processed} images in {elapsed:.1f}s "
          f"({total_processed / elapsed:.0f} img/s)")


if __name__ == "__main__":
    main()
