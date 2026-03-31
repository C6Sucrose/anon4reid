## ANON4REID

**Motivation:**  
Person re-identification (ReID) — matching people across different camera views — is useful for surveillance but conflicts with privacy rights under the GDPR. Anonymization techniques can protect identities, but they also degrade ReID performance. The key question is: **how much privacy can you gain before the system becomes useless?**

**Goal:**  
Benchmark the privacy–utility trade-off on the **Market-1501 dataset**: apply different anonymization techniques to person images and measure how much ReID accuracy drops at each level.

---

### Approach

*   **State-of-the-Art (SOTA) Research & Analysis:** Survey person ReID methods and existing privacy-preserving approaches (anonymization, identity shift, synthetic data).
*   **Establish Baseline:** Train or use a pre-trained ReID model (e.g., ResNet-50 with triplet loss, or OSNet) on the original Market-1501 dataset. Record baseline **mAP** and **Rank-1** accuracy.
*   **Anonymization Implementation:** Implement at least three anonymization techniques of increasing strength.
*   **Performance Testing:** Re-run the ReID model on each anonymized version of the dataset. Compare **mAP** and **CMC curves** across levels.
*   **Privacy Measurement:** Test whether a simple identity classifier trained on original data can still recognize the subjects. Visualize the trade-off (accuracy vs. anonymization strength) and document results.

---


### Project Details

*   **Skills:** Programming (Python), Computer Vision, Deep Learning (PyTorch), Image Processing
*   **Contact:** robert.aufschlaeger@th-deg.de

---

### Scope

* **Dataset**: Market-1501.

* **Baseline Model**: Implementing/fine-tuning an existing, well-documented architecture (ResNet-50 with triplet loss or OSNet).

* **Anonymization**: Implementing three distinct techniques of increasing strength (for instance Gaussian Blur, Facial/Body Pixelation, Silhouette Masking/Edge Detection).

* **Metrics**: mAP/CMC (Rank-1) for utility, and Accuracy of a secondary identity classifier for privacy.

* **Reporting**: Compiled Latex report documenting the methodology and the privacy-utility trade-off curve.

---

