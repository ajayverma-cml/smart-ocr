import streamlit as st
import numpy as np
from PIL import Image
from helper import OCRProcessor, summarize_ocr_result


st.set_page_config(page_title="Smart OCR", layout="centered")
st.title("Smart OCR - Extract Text from Images")


def perform_ocr(image_np):
    """Run OCR using Tesseract on preprocessed image."""
    ocr = OCRProcessor(image_np)
    ocr.preprocess_image()
    text = ocr.extract_text()
    return text

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or drag-and-drop here",
    type=["png", "jpg", "jpeg"]
)

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Show selected image
    st.subheader("Selected Image:")
    st.image(image_np, width='stretch')

    # Run OCR with loader
    with st.spinner("Processing image, please wait..."):
        text = perform_ocr(image_np)
        summary_data = summarize_ocr_result(text)
        st.subheader("OCR Result:")
        if summary_data:
            st.write(f"**Image Text**: {summary_data['refined_ocr_result']}")
            st.write(f"**Summary**: {summary_data['summary']}")
            st.write(f"**Language**: {summary_data['language']}")
            st.write(f"**Tone**: {summary_data['tone']}")
        else:
            st.text_area("", text, height=200)
