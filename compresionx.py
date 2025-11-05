import os
import time
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Compression and Upscaling Functions
def compression_algorithm(image_file, output_path, quality=40):
    try:
        img = Image.open(image_file)
        img.save(output_path, optimize=True, quality=quality)
    except FileNotFoundError:
        st.error("File not found. Please upload a valid image.")
    return output_path

# AI Upscaling using OpenCV

def ai_upscale_image(input_path, output_path, model_dir="models", model_name="EDSR_x4.pb", scale=4, max_width=540):
    """
    Upscale image using EDSR model (local .pb) while preserving original aspect ratio.
    max_width controls how big the input is resized before 4× upscaling.
    """
    try:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

        # Initialize model
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", scale)

        # Load image
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"❌ Could not open {input_path}")

        # Get original size and maintain aspect ratio
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Resize so width ≈ max_width while keeping ratio
        if w > h:
            new_w = max_width
            new_h = int(max_width / aspect_ratio)
        else:
            new_h = max_width
            new_w = int(max_width * aspect_ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Progress bar
        progress = st.progress(0)
        status_text = st.empty()
        for percent in range(0, 101, 10):
            progress.progress(percent / 100)
            status_text.text(f"🧠 Upscaling with EDSR... {percent}%")
            time.sleep(0.05)

        # Upscale 4×
        result = sr.upsample(img)
        cv2.imwrite(output_path, result)

        progress.empty()
        status_text.text(f"✅ Aspect ratio preserved — upscaled to {result.shape[1]}×{result.shape[0]}!")

        return output_path

    except Exception as e:
        st.error(f"Upscaling failed: {e}")
        return input_path
    
# Start of frontend
st.set_page_config(page_title="CompressionX", layout="centered")
st.sidebar.title("Options")
page = st.sidebar.radio("Go to", ["Home", "Demo", "About", "Contact"])

st.title("📦 CompressionX")

if page == "Home":
    st.title("Welcome to CompressionX")
    st.markdown("""Explore AI-powered compression and upscaling.  
    Upload an image to see the pipeline in action!""")

elif page == "Demo":
    st.subheader("🧠 Live Demo")
    st.markdown("""
    see our compression pipeline in action above.
    """)

    image_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    os.makedirs("temp", exist_ok=True)

    if image_upload:
        original_image_path = os.path.join("temp", image_upload.name)
        compressed_image_path = os.path.join("temp", "compressed_" + image_upload.name)
        ai_image_path = os.path.join("temp", "ai_upscaled_" + image_upload.name)

        img = Image.open(image_upload)
        img.save(original_image_path)

        compression_algorithm(original_image_path, compressed_image_path)
        compression_algorithm(compressed_image_path, compressed_image_path)
        ai_upscale_image(compressed_image_path, ai_image_path)

        st.image(Image.open(original_image_path), caption="Original Image", use_container_width=True)
        st.image(Image.open(compressed_image_path), caption="Compressed Image", use_container_width=True)
        st.image(Image.open(ai_image_path), caption="AI Upscaled Image", use_container_width=True)

elif page == "About":
    st.title("ℹ️ About CompressionX")
    st.markdown("""
    **CompressionX** explores AI-enhanced data compression.  
    We compress files at lower resolutions to reduce server stress  
    and then perform **AI upscaling locally** for real-time restoration.
    
    **Developer:** Richard Camacho  
    **Version:** 0.0.3
    """)

elif page == "Contact":
    st.title("📬 Contact Us")
    st.markdown("""
    For inquiries or collaboration:
    - **Developers:** Richard Camacho
    - **Email:** rcamacho11@ucmerced.edu
    - **Phone:** (951)581-0726
    """)

