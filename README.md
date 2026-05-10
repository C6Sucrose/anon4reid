# ANON4REID

**Motivation**

Person re-identification(matching people across different camera views) is essential for modern surveillance but creates a significant conflict with privacy rights under the **GDPR**. While anonymization can protect identities, it often degrades the features ReID models rely on. This project explores the boundary of this conflict: **How much privacy can be gained before the ReID system becomes functionally useless?**

**Goal**

Benchmark the privacy–utility trade-off on the **Market-1501 dataset** by applying four levels of anonymization, ranging from traditional blurring to generative AI synthesis, and measuring the resulting impact on ReID accuracy.

---

### Technical Approach

* **Baseline Establishment:** We utilize **OSNet (Omni-Scale Network)** as our primary ReID architecture. It is purpose-built for ReID, capturing both local and global features efficiently. We establish "Ground Truth" performance using pre-trained weights on the original Market-1501 dataset.
* **The Anonymization Pipeline:** We implement four levels of increasing privacy protection:
1. **Level 1 (Gaussian Blur):** Traditional obfuscation applied to head/face regions.
2. **Level 2 (AI-Powered Inpainting):** Using the **LaMa GAN** to semantically remove identifying features.
3. **Level 3 (RAD - Realistic Anonymization by Diffusion):** Using **Stable Diffusion + ControlNet (OpenPose)** to generate synthetic identities while preserving the original body pose and bounding box geometry.
4. **Level 4 (Edge Detection):** Applying **Canny Edge Detection** to strip all color and texture, representing the theoretical ceiling of privacy.


* **Resolution Handling:** To accommodate generative models (GAN/RAD) on the small **128x64** Market-1501 images, we implement an upscale-process-downscale pipeline (from 128x64 to 512x256 back to 128x64) to ensure high-quality synthetic generation without losing data compatibility.
* **Privacy Attack Simulation:** We train a secondary **Identity Classifier** (MobileNetV2) on original data to act as an "attacker", measuring how effectively each anonymization level thwarts identity recognition.

---

### Project Scope

* **Dataset:** Market-1501 (32,217 images, 1,501 identities).
* **Core Model:** **OSNet** (via `torchreid`) for utility measurement.
* **Generative Models:** LaMa GAN, Stable Diffusion v1.5 with LCM-LoRA and ControlNet.
* **Metrics:**
* **Utility:** mAP and CMC (Rank-1) curves.
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
3. **Sprint 3: Generation** — Batch processing of 32k+ images across distributed GPU clusters.
4. **Sprint 4: Evaluation** — Running ReID benchmarks and Privacy Attack simulations.
5. **Sprint 5: Synthesis** — Visualizing the trade-off curve and finalizing the report.

