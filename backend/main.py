import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add root folder to sys_path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.cv_utils import decode_image, encode_image, process_pipeline
from backend.models.segmentation import UNetSegmenter
from backend.models.classification import EfficientNetClassifier
from backend.models.explainability import GradCamExplainer

app = FastAPI(title="Skin Cancer AI Diagnosis API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
segmenter = UNetSegmenter()
classifier = EfficientNetClassifier()
explainer = GradCamExplainer()

@app.get("/")
def home():
    return {"message": "Skin Cancer AI Backend Active"}

@app.post("/analyze")
async def analyze_lesion(file: UploadFile = File(...)):
    """
    Main integrated pipeline: 
    Image -> Enhance -> Segment -> Classify -> Explain
    """
    try:
        # 1. Read Image
        contents = await file.read()
        image = decode_image(contents)
        
        # 2. Preprocessing & Enhancement
        enhanced_image = process_pipeline(image)
        
        # 3. Segmentation (U-Net)
        mask_binary = segmenter.predict(enhanced_image)
        segmented_overlay = segmenter.overlay_mask(enhanced_image, mask_binary)
        
        # 4. Classification (EfficientNet)
        classification_result = classifier.predict(enhanced_image, mask_binary)
        
        # 5. Explainable AI (Grad-CAM)
        heatmap_image = explainer.generate_heatmap(enhanced_image, mask_binary)
        
        # Prepare Response (Encode images to base64 for frontend consumption)
        def to_b64(img):
            return base64.b64encode(encode_image(img)).decode('ascii')
            
        return JSONResponse(content={
            "success": True,
            "prediction": classification_result["prediction"],
            "confidence": classification_result["confidence"],
            "probabilities": classification_result["probabilities"],
            "images": {
                "enhanced": to_b64(enhanced_image),
                "segmented": to_b64(segmented_overlay),
                "gradcam": to_b64(heatmap_image)
            }
        })
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.post("/dermatologist")
async def analyze_dermatologist(file: UploadFile = File(...)):
    """
    Direct Multimodal LLM pipeline for generic skin evaluations
    (cuts, burns, rashes, general dermatology) using Google Gemini Vision.
    """
    try:
        import os
        import google.generativeai as genai
        from PIL import Image
        import io
        
        # Dynamically read .env so it hot-reloads instantly without Uvicorn restart flags
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        api_key = None
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip().strip('"\'')
                        
        if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
            return JSONResponse(content={"success": False, "error": "API Key missing! Please open the .env file in the project folder and paste your key."})
        
        genai.configure(api_key=api_key)
        
        # Dynamically evaluate the user's API key to find an available Vision-capable model
        target_model = None
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name.lower() or 'vision' in m.name.lower() or '1.5' in m.name.lower():
                    target_model = m.name
                    if 'flash' in m.name.lower():
                        break # Prioritize flash for speed if available
                        
        if not target_model:
            raise Exception("Your current Google API Key does not have access to any Vision-capable Gemini models.")
            
        model = genai.GenerativeModel(target_model)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        prompt = """
        You are an expert AI Dermatologist. 
        A patient has uploaded this image of a skin condition (it could be a cut, burn, rash, infection, or general anomaly).
        Please analyze the image carefully and provide a very professional, medical-grade response structured as follows:
        1. **Observation**: What do you visually see in the image?
        2. **Potential Identification**: What is the most likely issue (e.g. 2nd-degree burn, eczema, abrasion)?
        3. **Medical Explanation**: What causes this condition?
        4. **Cures & Preventive Measures**: How should the patient heal this, keep it clean, or prevent further infection?
        5. **Doctor Consultation**: When is it critical to go see a real doctor?
        
        IMPORTANT: End your response with a disclaimer that you are an AI and this does not replace professional emergency medical diagnosis.
        """
        
        response = model.generate_content([prompt, image])
        return JSONResponse(content={"success": True, "analysis": response.text})
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
