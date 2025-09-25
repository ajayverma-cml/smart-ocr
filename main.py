import cv2
import pytesseract


class OCRProcessor:
    """
    OCRProcessor handles image preprocessing and text extraction
    using Tesseract OCR. It automatically adjusts text polarity.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)  # Original image
        self.processed_image = None  # Will hold preprocessed binary image

    def preprocess_image(self):
        """
        Convert image to grayscale, threshold it using Otsu method,
        and invert if necessary so text is dark on light background.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Binarize image using Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if text is white on black to match Tesseract preference
        if cv2.countNonZero(thresh) < (thresh.size // 2):
            thresh = 255 - thresh

        self.processed_image = thresh
        cv2.imwrite("preprocessed.png", self.processed_image)  # Save for debug/inspection

    def extract_text(self):
        """
        Extract text from the preprocessed image using Tesseract.
        Raises an error if preprocessing is not done.
        """
        if self.processed_image is None:
            raise ValueError("Image not preprocessed. Call preprocess_image() first.")

        # Use Tesseract to get detailed OCR output (word-level data)
        text = pytesseract.image_to_string(self.processed_image)
        return text


# Example usage
ocr = OCRProcessor("test.png")
ocr.preprocess_image()
extracted_text = ocr.extract_text()
print("Extracted Text:")
print(extracted_text.strip())
