import os
import json

import cv2
import pytesseract
from dotenv import load_dotenv

import google.generativeai as genai


load_dotenv()  # Load environment variables from .env file

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

class OCRProcessor:
    """
    OCRProcessor handles image preprocessing and text extraction
    using Tesseract OCR. It automatically adjusts text polarity.
    """
    def __init__(self, image_input):
        self.image_path = image_input
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input)
        else:
            # Assume it's a numpy array
            self.image = image_input
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

def parse_gemini_json(response_text):
    """
    Remove wrapping backticks and parse JSON from Gemini response.
    """
    # Remove ```json or ``` if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    # Remove optional "json" after opening ```
    cleaned = cleaned.lstrip("json").strip()

    # Parse JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def summarize_ocr_result(ocr_text):
    """
    Uses Gemini API to generate a concise, structured summary of the OCR result.

    Args:
        ocr_text (str): The text extracted from the image.

    Returns:
        str: JSON-formatted string containing:
            - refined_ocr_result: cleaned and corrected text
            - summary: brief explanation of the text
            - language: detected language
            - tone: overall tone of the text
    """
    try:
        # Initialize Gemini generative model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Improved prompt: focus on clarity, conciseness, structured JSON output
        prompt = f"""
You are a text processing assistant.

Task:
Analyze the following OCR text and generate a concise, readable summary.
- Clean any obvious OCR errors.
- Detect the language of the text.
- Detect the overall tone (e.g., Informative, Formal, Casual).

Output requirements:
- Return the result strictly in JSON format.
- JSON fields must be exactly:
  1. "refined_ocr_result": corrected OCR text
  2. "summary": brief explanation in plain English
  3. "language": detected language
  4. "tone": overall tone
- Do not include any explanation, markdown, or extra text outside JSON.

Example:
Input OCR Text:
"This is a smple text with errors."

Expected Output:
{{
  "refined_ocr_result": "This is a sample text with errors.",
  "summary": "A short text containing sample content with corrected errors.",
  "language": "English",
  "tone": "Informative"
}}

Now process the following OCR Text:
{ocr_text}
"""

        response = model.generate_content(prompt)
        parsed_json = parse_gemini_json(response.text.strip())
        return parsed_json
    except Exception as e:
        print(f"Summarization failed, Error: {str(e)}")
        return None
