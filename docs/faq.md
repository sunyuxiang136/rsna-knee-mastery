# Frequently Asked Questions (FAQ)

Answers to common questions regarding architecture, mathematical formulation, engineering choices,
and reproduction.

---

## 1. Modeling & Architecture

### Q: Why use 2.5D slice stacking instead of full 3D convolutions or 3D Vision Transformers?
**A:** Clinical knee MRI series are anisotropic: in-plane resolution is high (0.3 mm – 0.5 mm), while slice
thickness is often 3 mm – 4 mm. Furthermore, series slice counts vary widely between acquisition protocols.
- **3D Convolutions** require massive GPU VRAM, suffer from severe memory limitations at high spatial
  resolutions, and require complex 3D interpolation when slice spacing varies.
- **2D Networks with 3-Channel 2.5D Stacking** group 3 adjacent slices `(z-1, z, z+1)` into the RGB input
  channels of a standard pre-trained 2D backbone (e.g., CoAtNet, ConvNeXt). This captures local 3D through-plane
  context while allowing the model to leverage state-of-the-art 2D ImageNet pre-training at standard 224/384 px resolutions.

### Q: Why does each finding have its own attention vector (Class-Specific Attention)?
**A:** A knee MRI study contains 96 slices across 3 planes. Different diseases manifest on completely different
slices:
- An **ACL tear** is typically visible only on 2–3 specific sagittal slices.
- **Tibiofemoral osteoarthritis** is visible across multiple coronal slices showing joint space narrowing.
- A **Baker's cyst** appears in the posterior popliteal space on axial and sagittal views.

If a single global attention pooling layer were used, the network would be forced to find a single compromise set
of "important slices" for all 12 diseases simultaneously. By projecting window features through a `[F, 12]`
weight matrix (`class_attention`), each disease independently attends to the anatomical slices relevant to its
own diagnosis.

### Q: Why sample 2%–98% of the stack instead of just the central 15%–85%?
**A:** In standard computer vision, peripheral borders are often cropped or ignored. In knee anatomy, however, the
**medial and lateral collateral ligaments (MCL / LCL)** and the peripheral margins of the **lateral meniscus**
sit at the extreme outer edges of the coronal and sagittal acquisitions. Restricting sampling to the central
15%–85% measurably degrades validation AUC for collateral ligament injuries.

---

## 2. Ensembling & Post-Processing

### Q: Why use Percentile Rank Blending instead of simple probability averaging?
**A:**
1. **The Evaluation Metric (Macro-AUC)**: The Area Under the ROC Curve is purely a rank-order metric. It depends
   solely on whether positive cases are ranked higher than negative cases, completely invariant to monotonic score
   transformations.
2. **Model Calibration Discrepancies**: Different neural network architectures and loss functions produce vastly
   different output probability distributions. For example, a well-regularized model might output probabilities
   concentrated in `[0.40, 0.60]`, while another model outputs sharp predictions in `[0.05, 0.95]`. Averaging
   their raw probabilities causes the sharper model to completely dominate the ensemble, erasing the signal of the
   conservative model.
3. **Rank Normalization**: Converting raw predictions to empirical percentile ranks `rank(x) / N` projects all
   model outputs onto a uniform scale in `(0, 1]`, ensuring fair, proportional ensembling.

### Q: How does the pipeline defend against corrupted studies without crashing?
**A:**
- If an individual series or study fails preprocessing, the pipeline fills its prediction with a sentinel
  value of `0.5` rather than raising an uncaught exception.
- In Stage 3 fusion, the function `pct_rank_on(values, ok)` computes percentile ranks **strictly among non-failed
  studies**. Failed rows remain `NaN` during ranking and are then smoothly replaced by the alternate model's rank.
  This prevents artificial `0.5` sentinel values from distorting the ranks of correctly scored studies.
- If all models fail for a given study, the final submission safely outputs `0.5`, guaranteeing a valid,
  non-empty submission file.

---

## 3. Practical Usage & Environment

### Q: Can I run this repository without a GPU?
**A:** **Yes!**
- The entire `src/rsna_knee` package and all 22 unit tests (`pytest`) run in pure NumPy and standard Python.
- `make test` runs in under 0.1 seconds on any modern CPU.
- `make demo` (or `python examples/beginner_demo.py`) demonstrates all 4 algorithmic concepts in ~5 seconds with
  zero GPU requirements.

### Q: What is the difference between `src/rsna_knee/` and `reference/`?
**A:**
- `src/rsna_knee/`: Clean, tested, modularized Python package containing the core algorithmic logic
  (coordinate math, slot sampling, attention pooling, percentile rank fusion) under the **MIT License**.
- `reference/`: Raw, unedited Kaggle competition inference scripts extracted from public community notebooks.
  Included for transparency and exact reproducibility; governed by their upstream authors' licensing and terms
  (see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)).

### Q: Can this methodology be applied to other medical imaging tasks?
**A:** Yes. The three pillars of this methodology:
1. **Physical Millimetre Cropping** via DICOM voxel spacing tags;
2. **2.5D Multi-Instance Attention Pooling** with class-specific projections;
3. **Percentile-Rank Blending** with failure quarantine;
are directly applicable to any multi-sequence cross-sectional imaging domain, including **shoulder MRI,
lumbar spine MRI, hip arthrography, and multi-phase abdominal CT**.
