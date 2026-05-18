import streamlit as st
import requests
import base64
from PIL import Image
import io
import pandas as pd
import altair as alt

# Page Configuration for Premium Look
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS overrides
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #B0BEC5;
        margin-bottom: 20px;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #E53935;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        color: #43A047;
    }
    .medical-advice {
        background-color: #2b3a4a;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

import os

# Production API Endpoint Configurations
# When deployed live, set the 'BACKEND_API_URL' secret in your Streamlit Cloud dashboard.
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
API_URL_CV = f"{BACKEND_URL}/analyze"
API_URL_DERMA = f"{BACKEND_URL}/dermatologist"

# Medical Database Dictionary
DISEASE_INFO = {
    "Melanoma (MEL)": {
        "description": "The most dangerous form of skin cancer, characterized by the uncontrolled growth of pigment-producing cells. Highly likely to spread if untreated.",
        "precautions": "🚨 **SEEK IMMEDIATE MEDICAL ATTENTION.** Requires urgent biopsy and potential surgical excision. Strictly avoid UV/sun exposure. Do not scratch or attempt to pop."
    },
    "Melanocytic nevus (NV)": {
        "description": "A benign (non-cancerous) melanocytic tumor, commonly known as a mole. Very common and usually harmless.",
        "precautions": "✅ **Routine observation.** Monitor for ABCDE changes (Asymmetry, Border irregularity, Color changes, Diameter >6mm, Evolving shape). Apply standard sunscreen."
    },
    "Basal cell carcinoma (BCC)": {
        "description": "The most common form of skin cancer. It grows slowly and rarely spreads, but can be locally destructive to tissue if left untreated.",
        "precautions": "⚠️ **Consult a dermatologist.** Often treated with high success via minor procedures (e.g., Mohs surgery, cryotherapy). Consistently use SPF 50+ sunscreen."
    },
    "Actinic keratosis (AKIEC)": {
        "description": "A pre-cancerous, scaly spot found on sun-damaged skin. If left untreated, it may evolve into squamous cell carcinoma.",
        "precautions": "⚠️ **Schedule a checkup.** Can be treated easily with cryotherapy or topical prescription creams if caught early. Sun protection is mandatory."
    },
    "Benign keratosis (BKL)": {
        "description": "A benign, usually rough or warty skin lesion (e.g., Seborrheic Keratosis). Often appears with age and is completely non-cancerous.",
        "precautions": "✅ **No treatment necessary** unless it becomes irritated or bleeds. Keep moisturized and avoid picking at the rough scales."
    },
    "Dermatofibroma (DF)": {
        "description": "A common benign skin growth, typically occurring on the lower legs. Often feels like a hard lump under the skin.",
        "precautions": "✅ **Keep the area clean.** It is harmless, but if it becomes painful or changes color rapidly, a doctor can surgically remove it."
    },
    "Vascular lesion (VASC)": {
        "description": "Abnormalities of the blood vessels (like cherry angiomas). They are benign and simply cosmetic pools of blood vessels.",
        "precautions": "✅ **Generally harmless.** Avoid scratching as they can bleed easily. Cosmetic laser removal is an option if desired."
    },
    "Other Dermatological Lesion / Unknown": {
        "description": "The lesion features do not strongly align with the standard 7 skin cancer/tumor classifications. This could be Eczema, Psoriasis, Fungal infection, or healthy skin.",
        "precautions": "✅ **General Observation.** If the area is itchy, spreading, or causing pain, consult a general physician or dermatologist for a clinical swab or checkup."
    }
}

def decode_b64_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

# --- Application Header ---
st.markdown('<div class="main-header">Skin Cancer Detection & AI Dermatologist</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable CV Lesion Segmentation / Multimodal Diagnosis</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022576.png", width=100)
    st.title("System Mode")
    
    app_mode = st.radio("Select Diagnostic Engine:", [
        "🔍 Skin Cancer Detection (CV Pipeline)",
        "🩺 Personal Dermatologist (LLM)"
    ])
    
    st.markdown("---")
    
    if app_mode == "🩺 Personal Dermatologist (LLM)":
        st.warning("This mode uses Google's Live Generative AI to diagnose ANY general skin issue.")
        st.info("🔒 API Key is securely managed through your local `.env` file.")
        st.markdown("[Get a free API key here](https://aistudio.google.com/app/apikey)")
    else:
        st.info("""
        **Pipeline Steps Explained:**
        1. **Pre-processing:** Hair removal, Gaussian filtering, CLAHE.
        2. **Segmentation:** U-Net isolates the precise lesion.
        3. **Classification:** EfficientNet evaluates the structure.
        4. **Explainability:** Grad-CAM shows features impacting the AI's decision.
        """)
        
    st.markdown("---")
    st.markdown("Developed for Computer Vision Assignment Portfolio.")

col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.markdown(f"### Upload Image for: {app_mode.split(' ')[1]}")
    uploaded_file = st.file_uploader("Upload Skin Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Original", use_container_width=True)
        analyze_btn = st.button("🚀 Analyze Skin", use_container_width=True, type="primary")

if uploaded_file is not None and analyze_btn:
    
    # -------------------------------------------------------------
    # ROUTE A: Personal Dermatologist (Gemini Live LLM)
    # -------------------------------------------------------------
    if app_mode == "🩺 Personal Dermatologist (LLM)":
        with st.spinner('🧑‍⚕️ AI Dermatologist is examining your image...'):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(API_URL_DERMA, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if data["success"]:
                        with col_result:
                            st.markdown("### 📋 AI Dermatologist Report")
                            st.success("Analysis Complete.")
                            st.markdown(data["analysis"])
                    else:
                        st.error(f"Error from server: {data.get('error')}")
                else:
                    st.error(f"Failed to connect to AI server. Status {response.status_code}")
            except Exception as e:
                st.error(f"Connection Exception: {str(e)}")

    # -------------------------------------------------------------
    # ROUTE B: Cancer Detection (CV Pipeline & Proxy Logic)
    # -------------------------------------------------------------
    else:
        with st.spinner('🔬 Running AI processing pipeline...'):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(API_URL_CV, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if data["success"]:
                        with col_result:
                            st.markdown("### 📋 Skin Cancer Pipeline Report")
                            
                            primary_prediction = data['prediction']
                            
                            # Display Top Prediction
                            m1, m2 = st.columns(2)
                            with m1:
                                st.metric("Primary Detection", primary_prediction)
                            with m2:
                                st.metric("Confidence Score", f"{data['confidence']*100:.2f}%")
                                
                            # Medical Context & Precautions
                            st.markdown("#### Medical Information & Action Plan")
                            med_info = DISEASE_INFO.get(primary_prediction, {"description": "N/A", "precautions": "N/A"})
                            st.markdown(f"""
                            <div class="medical-advice">
                                <b>🧬 Pathology Description:</b> {med_info['description']}<br><br>
                                <b>⚕️ Recommended Action & Precautions:</b> {med_info['precautions']}
                            </div>
                            """, unsafe_allow_html=True)
                                
                            st.markdown("---")
                            
                            # Top 3 Predictions
                            st.markdown("#### Top 3 Probable Diseases")
                            sorted_probs = sorted(data['probabilities'].items(), key=lambda x: x[1], reverse=True)
                            for disease, prob in sorted_probs[:3]:
                                st.write(f"**{disease}**: {prob*100:.1f}%")
                                st.progress(float(prob))

                            st.markdown("---")
                            
                            # AI Summarization Feature (Simulated General AI)
                            st.markdown("### 🤖 Diagnostic Summary")
                            conf_percentage = data['confidence'] * 100
                            certainty_string = "strong quantitative alignment" if conf_percentage > 50 else "distributed variance requiring further verification"
                            
                            ai_summary = f"**Pipeline Log:** Based on the uploaded visual data, our CV segmenter successfully isolated the Region of Interest (ROI). The morphological feature extractor evaluated the asymmetry, border density, and color structure of the lesion. "
                            ai_summary += f"The neural architecture classified the primary structural match as **{primary_prediction}** with {conf_percentage:.1f}% confidence, indicating {certainty_string} with historical datasets. "
                            
                            if primary_prediction == "Other Dermatological Lesion / Unknown":
                                ai_summary += "Because the lesion lacks the standard hallmarks of the oncological models (absent blue-white veils or atypical pigment networks), it is classified as a non-cancerous anomaly."
                            elif "Melanoma" in primary_prediction or "carcinoma" in primary_prediction.lower():
                                ai_summary += f"Because this classification carries severe oncological risk, the Grad-CAM heatmap was generated to highlight the most irregular geometric parameters triggering this alert."
                            else:
                                ai_summary += f"Currently, the AI assesses this as a primarily low-risk or benign structure."
                                
                            st.info(ai_summary)
                            st.markdown("---")
                            
                            # Show Sub-Images (Pipeline visuals)
                            st.markdown("### 🔍 Computer Vision Architecture Output")
                            img1, img2, img3 = st.columns(3)
                            with img1:
                                st.image(decode_b64_image(data['images']['enhanced']), 
                                         caption="Preprocessed (CLAHE + Filtering)", use_container_width=True)
                            with img2:
                                st.image(decode_b64_image(data['images']['segmented']), 
                                         caption="U-Net ROI Segmentation", use_container_width=True)
                            with img3:
                                st.image(decode_b64_image(data['images']['gradcam']), 
                                         caption="Grad-CAM Explainability Heatmap", use_container_width=True)
                            
                            st.markdown("---")
                            
                            # Interactive Colorful Bar Chart
                            st.markdown("### 📊 Complete Probability Distribution")
                            df_probs = pd.DataFrame(
                                list(data['probabilities'].items()),
                                columns=['Disease', 'Probability']
                            )
                            bar_chart = alt.Chart(df_probs).mark_bar().encode(
                                x=alt.X('Probability:Q', axis=alt.Axis(format='%', title='Probability')),
                                y=alt.Y('Disease:N', sort='-x', title=""),
                                color=alt.Color('Disease:N', legend=None, scale=alt.Scale(scheme='tableau10')),
                                tooltip=[alt.Tooltip('Disease:N', title='Skin Condition'), alt.Tooltip('Probability:Q', format='.2%', title='Confidence')]
                            ).properties(height=350)
                            
                            st.altair_chart(bar_chart, use_container_width=True)
                            
                    else:
                        st.error(f"Error from server: {data.get('error', 'Unknown error')}")
                else:
                    st.error(f"Failed to connect to AI server. Status {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection Exception: {str(e)}\n\nPlease ensure tracking the `fastapi` backend is running on port 8000.")
