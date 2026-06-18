# ANON4REID

**Motivation**

Person re-identification(matching people across different camera views) is essential for modern surveillance but creates a significant conflict with privacy rights under the **GDPR**. While anonymization can protect identities, it often degrades the features ReID models rely on. This project explores the boundary of this conflict: **How much privacy can be gained before the ReID system becomes functionally useless?**

**Goal**

Benchmark the privacy–utility trade-off on the **Market-1501 dataset** by applying four levels of anonymization, ranging from traditional blurring to generative AI synthesis, and measuring the resulting impact on ReID accuracy.

---

### Technical Approach

* **Baseline Establishment:** We utilize **OSNet (Omni-Scale Network)** as our primary ReID architecture. It is purpose-built for ReID, capturing both local and global features efficiently. We establish "Ground Truth" performance using pre-trained weights on the original Market-1501 dataset.
* **The Anonymization Pipeline:** We implement four levels of increasing privacy protection:
1. **Level 1 (Gaussian Blur):** Gaussian blur applied to the upper portion of the bounding box (top 28%), approximating head/face obfuscation for tightly cropped pedestrian images.
2. **Level 2 (AI-Powered Inpainting):** Using **Stable Diffusion Inpainting** with a geometric mask covering the central torso region to remove identifying features.
3. **Level 3 (RAD - Realistic Anonymization by Diffusion):** Using **Stable Diffusion v1.5 + ControlNet (OpenPose)** accelerated with **LCM-LoRA** (8-step inference, trading quality for speed) to generate synthetic identities while preserving the original body pose and bounding box geometry.
4. **Level 4 (Edge Detection):** Applying **Canny Edge Detection** to strip all color and texture, representing the theoretical ceiling of privacy.


* **Resolution Handling:** To accommodate generative models (SD Inpainting/RAD) on the small **64×128** (width×height) Market-1501 images, we implement an upscale-process-downscale pipeline (from 64×128 to 256×512 back to 64×128) to ensure high-quality synthetic generation without losing data compatibility.
* **Privacy Attack Simulation:** We train a secondary **Identity Classifier** (MobileNetV2) on original data to act as an "attacker", measuring how effectively each anonymization level thwarts identity recognition.

---

### Project Scope

* **Dataset:** Market-1501 (36,036 images across 12,936 train + 19,732 test + 3,368 query splits, 1,501 identities).
* **Core Model:** **OSNet** (via `torchreid`) for utility measurement.
* **Generative Models:** Stable Diffusion Inpainting (sd-legacy/stable-diffusion-inpainting), Stable Diffusion v1.5 with LCM-LoRA and ControlNet.
* **Metrics:**
* **Utility:** mAP and CMC curves (Rank-1, 5, 10, 20).
* **Privacy:** Top-1 Accuracy of the Identity Classifier.


* **Deliverables:** A comprehensive privacy-utility trade-off curve and a LaTeX-compiled research report.

---

### Team & Skills

* **Skills:** Python, PyTorch, Computer Vision (OpenCV), Generative AI (Diffusers), Image Processing.
* **Authors:** [Huraira Ali](huraira.ali.prof@proton.me), Jazib Aslam, Kazi Shafwan Ur Rehman, Muhammad Umer Awan
* **Contact:** robert.aufschlaeger@th-deg.de

---

### Milestone Overview (Agile Kanban)

1. **Sprint 1: Baseline** — OSNet environment setup and original performance logging.
2. **Sprint 2: Pipeline** — Development of the 4-level anonymization scripts and resolution-fix logic.
3. **Sprint 3: Generation** — Batch processing of 36k+ images using Kaggle GPU notebooks.
4. **Sprint 4: Evaluation** — Running ReID benchmarks and Privacy Attack simulations.
5. **Sprint 5: Synthesis** — Visualizing the trade-off curve and finalizing the report.

