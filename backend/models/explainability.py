import cv2
import numpy as np

class GradCamExplainer:
    """
    Generates Explainable AI (XAI) visualizations.
    In a fully linked GPU environment, this would hook into the EfficientNet 
    final convolutional layer. For this real-time deployed dashboard, we 
    simulate the Grad-CAM behavior by generating a Gaussian activation 
    map centered on the most critical features (the lesion boundaries) 
    found during segmentation.
    """
    def __init__(self):
        pass
        
    def generate_heatmap(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Creates a mock Grad-CAM heatmap highlighting the lesion 
        using a jet color map simulation based on the mask distance transform.
        """
        if np.sum(mask) == 0:
            return image
            
        # Calculate distance transform to get the center of the lesion
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize the distance transform into a probability map (activation)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # We also want to emphasize the expanding border of the lesion
        # as melanoma is defined heavily by border irregularity.
        edges = cv2.Canny(mask, 100, 200)
        edges = cv2.GaussianBlur(edges, (21, 21), 0)
        cv2.normalize(edges, edges, 0, 0.6, cv2.NORM_MINMAX)
        
        # Combine center focus + border focus for the final activation map
        activation = np.clip(dist_transform + (edges.astype(np.float32) / 255.0), 0, 1)
        
        # Convert to 8-bit heatmap
        heatmap = np.uint8(255 * activation)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on the original image (0.4 intensity)
        superimposed_img = heatmap_colored * 0.4 + image * 0.6
        return superimposed_img.astype(np.uint8)
