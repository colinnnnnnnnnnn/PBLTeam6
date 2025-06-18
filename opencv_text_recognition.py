import cv2
import os
import argparse
import numpy as np

# Configure Tesseract path for Windows
try:
    import pytesseract
    if os.name == 'nt':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract found at: {path}")
                break
        else:
            print("Warning: Tesseract not found in common locations. Make sure it's in your PATH.")
except ImportError:
    print("Warning: pytesseract not installed. OCR functionality will be limited.")

# Custom model import
try:
    from inferenceModel import ImageToWordModel
    from mltu.configs import BaseModelConfigs
except ImportError:
    ImageToWordModel = None
    BaseModelConfigs = None

def ocr_whole_image(image_path, psm=6):
    import pytesseract
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=f'--psm {psm}')
    print("\nTesseract OCR (whole image):\n" + "-"*30)
    print(text.strip())
    return text.strip()

def custom_model_inference(image_path, config_path):
    if ImageToWordModel is None or BaseModelConfigs is None:
        print("Custom model code not available.")
        return
    configs = BaseModelConfigs.load(config_path)
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    # Preprocess: ensure 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    text = model.predict(image)
    print("\nCustom Model OCR (whole image):\n" + "-"*30)
    print(text.strip())
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Text Recognition (Tesseract or Custom Model)")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the image file")
    parser.add_argument("--whole-image", "-w", action="store_true", help="Run Tesseract OCR on the whole image")
    parser.add_argument("--custom-model", "-c", type=str, help="Path to custom model config.yaml (for custom model inference)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    if args.custom_model:
        custom_model_inference(args.image, args.custom_model)
        return

    # Default: Tesseract whole image
    ocr_whole_image(args.image, psm=6)

if __name__ == "__main__":
    main() 