# 🔬 AI-Powered Skin Cancer Detection & Personal Dermatologist

Welcome to the **Explainable AI-Based Skin Lesion Classification** project. This is a full-stack platform designed to process medical imagery. It is architected explicitly to be used in **Two Distinct Ways** depending on the compute power available to the user.

---

## ⚙️ The Two-Way Implementation Architecture

This project is built to handle both large-scale hospital deep learning and fast, local laptop ML deployments. 

### Path A: The Local Edge Deployment (What we are using now)
Since downloading 3 Gigabytes of medical images and running Deep Learning requires huge external Cloud GPUs, we use this mode for fast localized demonstrations on edge devices (laptops).
* **How it works:** We explicitly extract mathematical Computer Vision features (Hu Moments, Shape Geometry, Laplacian Texture Variance, and RGB Colors). 
* **The Model:** We trained a Classical ML **Random Forest Classifier** (`cv_ml_model.pkl`) using `scikit-learn` on these physical traits. The FastAPI backend loads this model to run instant off-line predictions.
* **Bonus (The Personal Dermatologist):** For this mode, we integrated **Google Gemini 1.5**. If you upload an image of a scrape, burn, or unknown rash (i.e., not a tumor), the system bypasses the Random Forest and securely streams the image to Google's LLM to provide holistic triage and first-aid recommendations.

### Path B: The Cloud Medical Deep Learning Training
For researchers attempting to deploy the ultimate clinical version of this software, they must use massive computational clusters.
* **How it works:** We have programmed pure **PyTorch** Architectures. 
* **The Process:** Researchers must download the 3GB **HAM10000 Database** from Kaggle, nest it inside a `data/HAM10000/` directory, and execute `training/train_classifier.py`.
* **The Model:** The script mounts the datasets through PyTorch Custom Dataloaders and fine-tunes a Deep Convolutional **EfficientNet** Neural Network over 50 epochs utilizing cloud compute (like Google Colab T4 GPUs). 

---

## 🚀 Project Workflow (Start to End)

Regardless of which architectural path you use, the application flow operates synchronously:

1. **Upload**: User uploads an image via the **Streamlit** dashboard.
2. **API Routing**: Streamlit securely streams the data via HTTP POST to the decoupled **FastAPI** backend.
3. **Image Preprocessing**: OpenCV mathematically strips body hair via Morphological Black-Hat filters and normalizes contrast using CLAHE.
4. **Segmentation**: Otsu's thresholding binarizes the skin and draws bounding perimeters explicitly around the lesion.
5. **Feature Extraction**: Mathematical features are extracted and parsed by the ML Model to generate predictions mapping to 8 dermatological classes.
6. **Explainable AI (XAI)**: A Grad-CAM heatmap is rendered using distance-transforms to highlight explicit pixel boundary anomalies that triggered the alert, guaranteeing interpretability for doctors.

---

## 🛠️ Installation & Execution

### 1. Environment Configurations
Create a hidden `.env` file in the root directory. Store your Gemini API Key here (required for the General LLM Dermatologist feature).
```env
GEMINI_API_KEY="your-api-key-here"
```
*(When deploying Live on the web, simply place this key in your cloud provider's Secret Manager rather than uploading the `.env` file).*

### 2. Dependencies
Install the required libraries locally:
```bash
pip install -r requirements.txt
```

### 3. Local Execution
You must launch the Backend API and Frontend UI concurrently:
```bash
# Terminal 1 - Launch Backend API (Defaults to port 8000)
python -m uvicorn backend.main:app --reload

# Terminal 2 - Launch Frontend Dashboard
python -m streamlit run frontend/app.py
```
*(Windows users may simply launch the `run.bat` auto-executor).*
