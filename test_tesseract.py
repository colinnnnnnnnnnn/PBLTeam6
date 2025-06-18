import os
import cv2
import numpy as np

# Configure Tesseract path for Windows
try:
    import pytesseract
    # Try to set the Tesseract path for Windows
    if os.name == 'nt':  # Windows
        # Common Tesseract installation paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✓ Tesseract found at: {path}")
                break
        else:
            print("✗ Warning: Tesseract not found in common locations. Make sure it's in your PATH.")
    else:
        print("✓ Tesseract path configuration completed (non-Windows)")
        
    # Test Tesseract version
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
    except Exception as e:
        print(f"✗ Error getting Tesseract version: {e}")
        
except ImportError:
    print("✗ Error: pytesseract not installed. Please run: pip install pytesseract")
    exit(1)

def test_ocr_on_simple_text():
    """Create a simple test image with text and try to OCR it"""
    print("\n--- Testing OCR on simple text ---")
    
    # Create a simple test image with text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add black text
    cv2.putText(img, "HELLO WORLD", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    try:
        # Try OCR
        text = pytesseract.image_to_string(gray, config='--psm 8')
        print(f"✓ OCR Result: '{text.strip()}'")
        
        if text.strip():
            print("✓ OCR is working correctly!")
            return True
        else:
            print("✗ OCR returned empty text")
            return False
            
    except Exception as e:
        print(f"✗ OCR Error: {e}")
        return False

def test_ocr_on_binary_image():
    """Test OCR on a binary (black and white) image"""
    print("\n--- Testing OCR on binary image ---")
    
    # Create a binary image with text
    img = np.ones((100, 300), dtype=np.uint8) * 255  # White background
    
    # Add black text
    cv2.putText(img, "TEST 123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    try:
        # Try OCR with different PSM modes
        for psm in [6, 7, 8, 13]:
            text = pytesseract.image_to_string(img, config=f'--psm {psm}')
            print(f"PSM {psm}: '{text.strip()}'")
            
        return True
        
    except Exception as e:
        print(f"✗ OCR Error: {e}")
        return False

if __name__ == "__main__":
    print("Tesseract OCR Test Script")
    print("=" * 30)
    
    # Test basic functionality
    test1 = test_ocr_on_simple_text()
    test2 = test_ocr_on_binary_image()
    
    print("\n" + "=" * 30)
    if test1 and test2:
        print("✓ All tests passed! Tesseract OCR is working correctly.")
        print("You can now use the OpenCV text recognition scripts.")
    else:
        print("✗ Some tests failed. Please check your Tesseract installation.")
        print("Make sure Tesseract is properly installed and in your PATH.") 