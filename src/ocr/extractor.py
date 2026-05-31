"""OCR pipeline for extracting text from food label images."""

import easyocr
import cv2
from pathlib import Path

class LabelExtractor:
    def __init__(self, languages: list[str] = ['en']):
        print("Loading OCR models (this may take a moment on first run)...")
        self.reader = easyocr.Reader(languages, gpu=False) # Change to True if dedicated GPU available

    def extract_text(self, image_path: str | Path) -> str:
        """
        Extract text from an image. 
        Applies basic preprocessing (grayscale, contrast) to improve accuracy.
        """
        img_path = str(image_path)
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Load image with OpenCV
        image = cv2.imread(img_path)
        
        # Convert to grayscale (improves OCR contrast)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply basic thresholding/contrast (optional but helps noisy labels)
        # alpha=1.5 (contrast), beta=0 (brightness)
        adjusted = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Extract text
        print(f"Scanning image: {Path(img_path).name} ...")
        results = self.reader.readtext(adjusted, detail=0, paragraph=True)
        
        # Join all extracted text blocks into a single string
        full_text = " ".join(results)
        return full_text
