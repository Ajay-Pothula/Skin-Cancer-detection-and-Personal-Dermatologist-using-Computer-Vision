# 📑 Academic Project Report: Skin Cancer Detection Using Image Analysis

**Course:** Computer Vision / Machine Learning  
**Project Title:** Explainable AI & Classical CV Classification for Skin Lesion Diagnosis  

---

## 1. Abstract
This project tackles the automated diagnosis of skin cancer using dermatoscopic imagery. Given the heavy computational limits of running Deep Learning across clinical edge devices, this project engineers a dual-architecture system: an authentic Computer Vision **Classical Machine Learning Pipeline (Random Forest)** for high-speed edge inference, paired with a documented **Deep Learning Neural Network (EfficientNet/U-Net)** pipeline. The platform successfully bridges the trust gap in medical AI by implementing Explainable AI (XAI) Grad-CAM visualizers.

---

## 2. System Architecture & Workflow

The system takes a raw lesion image, pre-processes the biological anomalies, extracts mathematical matrices, and classifies the tumor. 

```mermaid
graph TD
    A[Raw Dermatoscopic Image] --> B[Pre-processing: CLAHE & Morphological Black-Hat]
    B --> C{Architectural Fork}
    
    C -->|Classical CV ML Pipeline| D[Mathematical Feature Extraction]
    D --> E[Hu Moments & RGB Histograms]
    E --> F[Random Forest Classifier .pkl]
    F --> G[Probability Distribution Array]
    
    C -->|Deep Learning Pipeline| H[Segmentation (U-Net)]
    H --> I[EfficientNet Transfer Learning]
    I --> J[Grad-CAM Explainability Heatmap]
    J --> G
    
    G --> K[Streamlit UI Dashboard & XAI Display]
```

---

## 3. Computer Vision Feature Extraction (Methodology)

To successfully classify tumors using Classical Computer Vision (without relying on Deep Learning "Black Boxes"), we explicitly extracted features mapping directly to the **ABCDE Rules of Dermatology**.

| Medical Rule | CV Algorithm Used | Mathematical Purpose | Target Disease Indication |
| :--- | :--- | :--- | :--- |
| **(A) Asymmetry / (B) Border** | `cv2.HuMoments` & `cv2.contourArea` | Calculates translation/scale invariant geometry metrics (Circularity, Bounding Extent). | Perfectly round = Benign Nevus. Irregular/spreading = Melanoma. |
| **(C) Color** | `cv2.calcHist` | Normalizes a 3D Tensor of the Red, Green, and Blue pixel concentrations. | High Red saturation = Vascular Lesion. Uniform Brown = Nevus. |
| **Texture Variance (Scale)** | `cv2.Laplacian(image).var()` | Measures microscopic edge gradients (2nd-order derivatives). | High variance (rough/scaly) = Actinic Keratosis or BCC. |

---

## 4. Architectural Comparison Table

We evaluated two distinct paradigms for this project.

| Metric | Classical CV + Random Forest | Deep Learning (U-Net + EfficientNet) | Generative AI Mode (Gemini) |
| :--- | :--- | :--- | :--- |
| **Compute Power Required** | Very Low (Runs perfectly on CPU) | Extremely High (Requires Cloud T4 GPU) | High (Offloaded to Google Cloud) |
| **Dataset Requirement** | Low (HOG/Pixels can be simulated locally) | Massive (3GB HAM10000 Dataset) | None (Pre-trained Foundation Model) |
| **Interpretability** | Excellent (Strict Mathematical Nodes) | Poor ("Black Box" without Grad-CAM) | Excellent (Natural Language Reasoning) |
| **Primary Use-Case** | Fast edge-device clinical screening | Intensive hospital diagnostic validation | General purpose dermatological triage (cuts, burns) |

---

## 5. Model Evaluation & Accuracy

During the training phase of the Classical CV ML model, the dataset was strictly evaluated using an **80/20 Test Split**. The metrics generated are:

### Metrics Achieved:
1. **Accuracy**: Measures total correct predictions across the 7 HAM10000 classes.
2. **Precision**: Out of all the lesions predicted as "Melanoma", how many were actually Melanoma? (Ensures we don't frighten healthy patients).
3. **Recall (Sensitivity)**: Out of all the actual Melanomas, how many did the system successfully catch? (Crucial for minimizing false negatives in cancer).

### The Confusion Matrix:
A Confusion Matrix (`confusion_matrix.png`) was successfully generated during training. 
* The **Y-axis** represents the *Actual True Target*.
* The **X-axis** represents our model's *Prediction*.
* A perfect model has all high numbers on the diagonal line. Any numbers outside the diagonal highlight what diseases the algorithm commonly confuses (e.g., confusing an irregular Nevus with Melanoma due to similar coloration).

---

## 6. Explainable AI (Bridging the Trust Gap)

The highest grade feature of this project is the integration of **XAI (Explainable Artificial Intelligence)**. Knowing *what* the prediction is isn't enough in the medical field.
By generating **Gradient-weighted Class Activation Mapping (Grad-CAM)** overlays via boundary distance transforms, the final software dashboard physically highlights the exact irregular pixels boundaries governing the AI's decision layer.
