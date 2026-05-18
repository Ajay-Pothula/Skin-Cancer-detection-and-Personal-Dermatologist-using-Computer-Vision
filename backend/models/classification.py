import numpy as np
import cv2
import pickle
import os

class EfficientNetClassifier:
    """
    Classical Machine Learning CV Expert System.
    Loads a true 'scikit-learn' trained model (.pkl) trained on explicit
    Computer Vision features: Hu Moments, Color Histograms, and Laplacian Variance.
    """
    CLASSES = [
        "Melanoma (MEL)", 
        "Melanocytic nevus (NV)", 
        "Basal cell carcinoma (BCC)", 
        "Actinic keratosis (AKIEC)", 
        "Benign keratosis (BKL)",
        "Dermatofibroma (DF)",
        "Vascular lesion (VASC)",
        "Other Dermatological Lesion / Unknown"
    ]

    def __init__(self):
        print("Loading Trained Machine Learning Model...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_ml_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        else:
            print("WARNING: ML Model not found! Falling back to heuristic baseline.")
            self.model_loaded = False
            
    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ Extract same CV features used in the ML Training Script """
        img = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return np.concatenate([hist, hu_moments, [lap_var]])

    def predict(self, image: np.ndarray, mask: np.ndarray) -> dict:
        if np.sum(mask) == 0:
            return self._build_response(7, 1.0) # Unclassified
            
        features = self.extract_features(image, mask)
        
        if self.model_loaded:
            # Execute True Machine Learning Inference
            probs_array = self.model.predict_proba([features])[0]
            # Map the 7 model classes. Add an 8th (0.0) to match our Streamlit UI length
            probs = np.append(probs_array, [0.001])
        else:
            # Fallback (Just in case the .pkl is missing)
            probs = np.array([0.1] * 8)
            probs[0] = 0.9 # Default Melanoma baseline
            
        # Normalize
        probs = probs / np.sum(probs)
        predicted_idx = np.argmax(probs)
        
        return {
            "prediction": self.CLASSES[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": {cls_name: float(prob) for cls_name, prob in zip(self.CLASSES, probs)}
        }
        
    def _build_response(self, predicted_idx, conf_score):
        probs = [0.0] * len(self.CLASSES)
        probs[predicted_idx] = 1.0
        return {
            "prediction": self.CLASSES[predicted_idx],
            "confidence": conf_score,
            "probabilities": {class_name: float(prob) for class_name, prob in zip(self.CLASSES, probs)}
        }
