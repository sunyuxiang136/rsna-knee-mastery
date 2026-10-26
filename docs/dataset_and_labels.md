# Dataset & Weak Supervision Label Engineering

This document outlines the clinical context, imaging characteristics, and weak-supervision label
extraction methodology used in the RSNA Knee Abnormality Detection challenge.

---

## 1. The Clinical Problem & Target Findings

Knee magnetic resonance imaging (MRI) is the gold standard non-invasive modality for assessing
musculoskeletal pathology. In this competition, algorithms must evaluate an entire multi-sequence
examination and output predicted probabilities for **12 distinct clinical findings**:

| Pathology | Target Compartment | Clinical Presentation | Primary Diagnostic Plane |
|---|---|---|---|
| **ACL** | Cruciate Ligament | Anterior cruciate ligament tear or rupture | Sagittal (liquid-sensitive / T2) |
| **MCL** | Collateral Ligament | Medial collateral ligament sprain or tear | Coronal (fluid-sensitive) |
| **Medial Meniscus** | Meniscal Cartilage | Medial fibrocartilage tear (root, body, horn) | Sagittal & Coronal |
| **Lateral Meniscus** | Meniscal Cartilage | Lateral fibrocartilage tear | Sagittal & Coronal (outer slices) |
| **Medial OA** | Articular Cartilage / Bone | Osteoarthritis in medial tibiofemoral joint | Coronal (anatomical / non-fluid) |
| **Lateral OA** | Articular Cartilage / Bone | Osteoarthritis in lateral tibiofemoral joint | Coronal (anatomical / non-fluid) |
| **PF OA** | Patellofemoral Joint | Osteoarthritis between patella and femoral trochlea | Axial & Sagittal |
| **Effusion** | Joint Capsule | Excess synovial fluid in suprapatellar bursa | Sagittal (T2-weighted fluid sensitive) |
| **Synovitis** | Synovial Membrane | Inflammation/thickening of synovial lining | Sagittal & Axial |
| **Baker's** | Popliteal Fossa | Baker's cyst (fluid distension in popliteal space) | Sagittal & Axial |
| **Contusion** | Trabecular Bone | Bone marrow edema / micro-fracture | Sagittal (fluid-sensitive / STIR) |
| **Fracture** | Cortical / Subchondral Bone | Traumatic bone fracture or impaction | Coronal & Sagittal |

---

## 2. Imaging Heterogeneity in Clinical MRI

Unlike standard 2D computer vision benchmarks where images are normalized RGB photographs, clinical knee
MRIs present three major structural hurdles:

### 2.1 Multi-Plane Series Composition
A complete knee study consists of multiple DICOM acquisition series:
- **Sagittal plane**: Views the knee from the lateral/medial side; critical for ACL, PCL, and meniscal horns.
- **Coronal plane**: Views the knee from the front/back; optimal for MCL, LCL, and tibiofemoral joint cartilage.
- **Axial plane**: Horizontal cross-sections; optimal for patellofemoral tracking, joint effusion, and Baker's cysts.

Different hospital scanners acquire varying numbers of series per patient, and series slice counts range
anywhere from 15 to over 60 slices.

### 2.2 Sequence Contrast (Fluid-Sensitive vs. Anatomical)
- **Fluid-Sensitive (e.g., T2-weighted, Proton Density with Fat Suppression - PD FS)**: Water and inflammation
  appear bright white. Essential for spotting fluid effusion, marrow edema, ligament tears, and cyst formation.
- **Anatomical (e.g., T1-weighted, non-fat-suppressed)**: Sharp tissue boundaries and bone marrow fat appear bright.
  Essential for evaluating subchondral bone remodeling, osteophytes, and cartilage thinning.

### 2.3 Physical Resolution vs. Pixel Dimensions
Clinical scanner pixel spacing varies between **0.3 mm** and **0.5 mm** per pixel.
- Direct image resizing to a fixed pixel count (e.g., $224 \times 224$) forces the neural network to conflate
  scanner voxel spacing with patient anatomy size.
- The pipeline solves this via **millimetre-based physical cropping** (`CROP_MM = 140.0 mm`), computing the exact
  box dimension from the DICOM `PixelSpacing` tag before resizing (see [`src/rsna_knee/geometry.py`](../src/rsna_knee/geometry.py)).

---

## 3. Weak Supervision: Breaking the Label Bottleneck

### 3.1 The Scarcity Challenge
- Total cohort: **4,407 studies**.
- Ground-truth expert annotations: **only 58 studies**!
- Remaining 4,349 studies: Accompanied strictly by **unstructured, free-text radiology reports**.

### 3.2 LLM Soft Label Extraction
Training deep neural networks directly on 58 samples leads to immediate catastrophic overfitting.
The solution pioneered in the competition literature involves reading the free-text narrative reports
with large language models (LLMs) to infer continuous probability targets:

1. **Continuous Confidences, Not Binary 0/1**:
   When a radiologist writes *"probable low-grade sprain of the anterior cruciate ligament"*, forcing a hard
   label of `1` or `0` introduces harmful label noise. Instead, the language model outputs a calibrated soft
   probability (e.g., `0.75`), preserving diagnostic uncertainty.
2. **Label Aggregation**:
   The training labels behind the reference weights combined outputs from several public LLM extractions
   (stevenleehans, pilkwang, and riadmohamed42's verified JEV labels) alongside OAI (Osteoarthritis Initiative)
   controlled-access reference data.

### 3.3 The Held-Out Gold Standard
Critically, the **58 expert-annotated gold studies were never included in the training set**. They were
strictly preserved as an untainted validation anchor. On these 58 gold cases, the single-model Raptor-224
reaches approximately **0.9167 macro-AUC**, demonstrating strong generalization from weakly supervised
reports to confirmed clinical ground truth.

---

## 4. Key Takeaways for Medical ML Engineering

1. **Physical Scaling Invariance**: In medical imaging, always normalize spatially by physical dimensions (mm)
   rather than native matrix pixel counts.
2. **Pathology-Informed Slicing**: Avoid sampling only central slices (e.g., 15%–85%); sampling outer slices
   (2%–98%) is necessary to capture peripheral anatomy like the collateral ligaments.
3. **Probabilistic Supervision**: When human expert labeling is cost-prohibitive, LLM-extracted soft targets
   from clinical notes provide a powerful, noise-tolerant training signal.
