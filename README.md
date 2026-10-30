# NeuroLens: Multimodal Alzheimer's Disease Classification via 3D Swin Transformers & Clinical Feature Fusion

NeuroLens is an advanced, high-performance deep learning framework engineered for the multimodal classification of Alzheimer's Disease (AD) progression stages using T1-weighted 3D Magnetic Resonance Imaging (MRI) and clinical/demographic profiles. Built upon the OASIS-1 (Open Access Series of Imaging Studies) dataset, the architecture fuses a cutting-edge **3D Swin Transformer** (Swin UNETR encoder) with an optional dual auxiliary spatial representation branch (**ConvNeXt3D** or **Multi-Scan Mamba**) and a **Clinical Multi-Layer Perceptron (MLP)**.

Our system classifies subjects into three clinical cohorts based on the Clinical Dementia Rating (CDR) scale:
1. **Cognitively Normal (CN)** (CDR = 0)
2. **Very Mild Dementia** (CDR = 0.5 - proxy for Mild Cognitive Impairment)
3. **Alzheimer's Disease (AD)** (CDR >= 1)

---

## 🛠️ Complete Project Directory Structure

```directory
NeuroLens/
├── configs/
│   └── default.yaml         # Centralized hyperparameters, path mappings, and network states
├── src/
│   ├── __init__.py          # Marks src/ directory as a Python package
│   ├── utils.py             # Shared evaluation metrics, configurations, and checkpointing helpers
│   ├── preprocess.py        # Volumetric resampling (96x96x96), skull masking, and intensity z-scoring
│   ├── dataset.py           # Subject-level Stratified K-Fold splitter, Clinical Preprocessor, and 3D Augmentation
│   ├── model.py             # Hybrid multimodal fusion network (Swin, ConvNeXt3D, Mamba, Clinical MLP)
│   ├── train.py             # Early-stopping, AMP-accelerated training loops, and validation/test folds
│   ├── evaluate.py          # Multiclass Macro-F1, Multi-threshold ROC-AUC, and Confusion Matrix compute
│   └── baseline.py          # Clinical-only baselines (Logistic Regression & Gradient Boosting Classifier)
├── README.md                # Comprehensive project documentation and benchmark guide
└── requirements.txt         # Core dependencies list
```

---

## 🚀 Multimodal Pipeline & Data Flow Architecture

Our end-to-end framework implements a highly robust pipeline from raw, unevenly-distributed neuroimaging scans to final localized spatial and tabular explanations.

```
[3D Raw Analyze MRI (.img)] ────► [Preprocessing] ────► [Voxel Resampling (96x96x96) & Z-Score] ───┐
                                                                                              │
[Clinical Metadata (.xlsx)] ───► [Subject Validation] ──► [Tabular Median-Imputation & Scale] ───┼─► [Multimodal Fusion Model] ─► [CDR Multi-class Logits]
                                                                                              │
[Swin_Second_Clinical Model] ◄── [Dual-Branch Feature Aggregation] ◄──────────────────────────┘
```

### 1. 3D MRI Volumetric Preprocessing
- **Resampling**: Translates various input spaces to a uniform isotropic volumetric grid of $96 \times 96 \times 96$ via trilinear interpolation.
- **Brain Masking**: Leverages pre-computed intracranial masks (`*_masked_gfc.img`) to discard uninformative non-brain background voxels, maximizing spatial-feature efficiency.
- **Intensity Normalization**: Calculates localized voxel statistics across brain tissue regions to scale intensities via zero-mean unit-variance $Z$-scoring, preventing training instability from scanner bias.
- **Storage Conservation**: Exports processed tensor cubes as compact 16-bit half-precision Float arrays (`.npy`), ensuring rapid loading cycles.

### 2. Clinical Profile Transformation
Tabular descriptors are handled in a specialized pipeline engineered to strictly avoid data leakage during cross-validation:
- **Features Standardized**: Continuous (Age, Educ, SES, eTIV, nWBV, ASF) and categorical (Gender, Handedness).
- **Median-Imputation**: Dynamically fills missing categorical elements (SES, Educ) based on training partitions.
- **Standardization**: Computes continuous features means and standard deviations from training splits only, projecting test subjects into an identical standardized feature space.
- **Missing Flags Generation**: Creates boolean indicators to represent missing values explicitly, preserving the latent signal of unrecorded clinical interactions.

### 3. 3D Volume Spatial Augmentations
Applied dynamically to the training loader to counter small-sample overfitting:
- **Axis Flipping**: Performs random horizontal reflections ($p=0.5$).
- **3D Affine Transformations**: Small random spatial rotations (up to 5 degrees) and 3D translations (up to 4% offset) computed via 3D affine matrices and trilinear grid sampling.
- **Intensity Jittering**: Random scaling factors alongside additive Gaussian white noise ($\sigma=0.03$) within target tissues.

---

## 🧠 Advanced Hybrid Model Architecture

NeuroLens supports a variety of modular backbones that merge independent feature extractors before projecting to the final classification heads:

1. **3D Swin Transformer Branch**: Integrates a Monai-powered 3D Swin Transformer (Swin UNETR encoder). By computing self-attention across non-overlapping shifted window grids, it captures multi-scale hierarchical context across brain volumes with linear computational complexity.
2. **ConvNeXt3D Convolutional Branch**: An auxiliary 3D CNN utilizing deep separable convolutional layers (depthwise $7 \times 7 \times 7$ kernels) and Layer Normalization. It extracts local shift-invariant structural features of neurodegeneration.
3. **3D Multi-Scan Mamba Branch**: An experimental alternative utilizing bidirectional State Space Models (SSMs). Images are tokenized into sequences and scanned across three different projection axes forwards and backwards, reconstructing long-range continuous context without attention bottlenecks.
4. **Clinical MLP Branch**: A specialized tabular network with hidden feedforward projections and Dropout regulation.
5. **Multimodal Late-Fusion Trunk**: Concatenates output features of active modules ($128_{\text{Swin}} + 128_{\text{ConvNeXt}} + 32_{\text{Clinical}} = 288$), funneling them through a regularized fully-connected multi-layer network before final soft-max categorization.

---

## 📊 Empirical Benchmarks & Ablation Performance

Validation is conducted using a rigorous subject-level Stratified 3-Fold Cross-Validation strategy to guarantee that scans from the same subject never overlap between partitions.

### Ablation Summary Table (OASIS-1 Sample)

| Model Backbone Config | Test Accuracy | Pooled Macro-F1 | Pooled Macro-AUC | Fold F1 Mean | Fold F1 Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical-only (Logistic Regression)** | 0.476 | 0.447 | 0.634 | 0.447 | 0.186 |
| **Clinical-only (Gradient Boosting)** | 0.524 | 0.404 | 0.716 | 0.405 | 0.176 |
| **Swin Transformer (Only)** | 0.524 | 0.395 | 0.613 | 0.354 | 0.114 |
| **ConvNeXt3D Convolutional (Only)** | 0.571 | 0.413 | 0.547 | 0.417 | 0.153 |
| **Swin + ConvNeXt3D Fusion** | 0.619 | 0.468 | 0.619 | 0.442 | 0.184 |
| **Swin + ConvNeXt3D + Clinical MLP** | **0.667** | **0.542** | **0.781** | **0.498** | **0.189** |

*Note: Obtained on a validation pipeline subset (~25 subjects) ensuring programmatic functional integrity across multi-gigabyte files prior to full-scale cloud compute training.*

---

## 🔍 Explainable AI (XAI) & Interpretability

Rather than presenting a "black-box" classification, NeuroLens provides two deep interpretability layers to validate clinical decisions:

### 1. Tabular Shapley Additive Explanations (SHAP)
Using game-theoretic attribution vectors on the clinical preprocessors, we map the exact directional force of each variable on the cohort classes:

- **Cognitively Normal (CN)**: High brain volumes (`nWBV`) and high functional performance scores (`MMSE`) act as strong positive attributions for healthy classifications.
- **Alzheimer's Disease (AD)**: Reduced standardized whole brain volumes, advanced age, and lower socioeconomic indicators consistently push the model away from CN towards AD category limits.

### 2. 3D Spatial Grad-CAM Localization
By tracking class-specific gradients back to the final layers of our 3D convolutional architectures, NeuroLens generates localized activation overlays directly onto raw patient spaces:

- **Target Layer**: `model.second.stages[-1]` (ConvNeXt3D block).
- **High Activation Regions**: Identifies critical structural patterns, highlighting localized voxel networks (e.g., surrounding the hippocampus and ventricular spaces) that influenced the target dementia predictions.

---

## 🖼️ Diagnostic and Interpretability Visualizations

Below are key diagnostic, preprocessing, and model explainability visualizations generated during our empirical pipeline runs.

### 1. Labelled Subject Cohort Distribution
Shows the structural balance across Cognitively Normal (CN), Very Mild Dementia (CDR 0.5), and Alzheimer's Disease (AD) classes within the pipeline-verification subset.

<p align="center">
  <img src="/content/Screenshot 2026-10-30 141612.png" width="650" alt="Cohort Class Distribution">
</p>

### 2. Isotropic Volumetric MRI Preprocessing Views
Illustrates central slices across the Sagittal, Coronal, and Axial planes after voxel resampling and $Z$-score tissue intensity normalization.

<p align="center">
  <img src="/content/Screenshot 2026-10-30 141634.png" width="850" alt="Processed T1 MRI Slices">
</p>

### 3. Unified Clinical Feature Distribution After Normalization
Distributions of all ten clinical markers after dynamic training-set median imputation and standardization (ensuring Mean $\approx 0$ and Std $\approx 1$).

<p align="center">
  <img src="/content/Screenshot 2026-10-30 141704.png" width="850" alt="Clinical Feature Normalizations">
</p>

### 4. Permutation Feature Importance Analysis
Quantifies predictive feature degradation (decrease in out-of-fold Macro-F1 score) when clinical attributes are independently permuted.

<p align="center">
  <img src="/content/Screenshot 2026-10-30 141744.png" width="650" alt="Permutation Importance">
</p>

### 5. Multi-Class Game-Theoretic SHAP Attributions
Visualizes local attribute forces pushing individual subject predictions towards CN (Cognitively Normal) or Very Mild Dementia statuses.

<p align="center">
  <img src="/content/Screenshot 2026-10-30 141749.png" width="550" alt="CN SHAP Beeswarm">
  <img src="/content/Screenshot 2026-10-30 141753.png" width="550" alt="Very Mild SHAP Beeswarm">
</p>

---

## 🛠️ Step-by-Step Execution Guide

### 1. Installation
Ensure you have CUDA-supported PyTorch installed, then install package prerequisites:
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
Execute the preprocessing script to resample raw structural volumes to the isotropic 96px resolution:
```bash
python -m src.preprocess
```

### 3. Run Clinical Baselines & Main Ablation
Configure parameters inside `configs/default.yaml` and kick off model cross-validation:
```python
from src.train import run_ablation
# Executing training across all specified ablation modes
result_df = run_ablation(cfg, df_valid, folds)
print(result_df)
```