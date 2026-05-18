import cv2
import numpy as np
from PIL import Image
import io

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw uploaded bytes to an OpenCV image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def encode_image(image: np.ndarray) -> bytes:
    """Encodes OpenCV image to PNG bytes."""
    _, buffer = cv2.imencode('.png', image)
    return buffer.tobytes()

def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Applies morphological transformations to remove hair artifacts
    from the dermoscopic lesion image. This is a crucial classical 
    Computer Vision step for skin cancer detection.
    """
    # Convert image to grayscale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Kernel for morphologyEx
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    
    # Apply Black-Hat filtering to isolate hairs (dark thin structures on lighter background)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # Intensify hair features to create a strong mask
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the hair mask to replace hairs with skin tone
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization 
    to enhance the contrast of the lesion features (border, color).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge back
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply slight median blur to remove salt-and-pepper noise post-CLAHE
    final = cv2.medianBlur(final, 3)
    return final

def process_pipeline(image: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """Full CV image preprocessing pipeline."""
    # 1. Resize for network consistency
    resized = cv2.resize(image, target_size)
    
    # 2. Artifact (Hair) Removal
    hair_removed = remove_hair(resized)
    
    # 3. Enhanced visualization (CLAHE + Noise removal)
    enhanced = apply_clahe(hair_removed)
    
    return enhanced
