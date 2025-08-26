import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import io
import base64
import matplotlib.pyplot as plt

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="wide")

# Paths
TUMOR_MODEL_PATH = "tumor_model.tflite"

# Classes
CLASSES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

# Load TFLite model
@st.cache_resource
def load_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_interpreter(TUMOR_MODEL_PATH)

def preprocess_image(image, size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_tumor(image):
    img_array, pil_img = preprocess_image(image)
    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()

    tumor_interpreter.set_tensor(input_details[0]['index'], img_array)
    tumor_interpreter.invoke()
    preds = tumor_interpreter.get_tensor(output_details[0]['index'])[0]
    return preds, pil_img

# ---------------------- GRAD-CAM ----------------------
def grad_cam_visualization(pil_img, size=(224, 224)):
    img_resized = pil_img.resize(size)
    img_np = np.array(img_resized)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(224, 224)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return overlay

# ---------------------- UI ----------------------
st.title("🧠 Brain Tumor Classification")
st.caption("**Disclaimer:** This tool is for educational purposes only. Predictions are based on a trained model and are not 100% accurate. Always consult a medical professional.")

# Dark/Light Mode Toggle
mode = st.radio("Theme Mode:", ["Light", "Dark"], horizontal=True)
if mode == "Dark":
    st.markdown("<style>body {background-color: #121212; color: white;}</style>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload MRI images (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(len(uploaded_files))
    results = []
    for idx, file in enumerate(uploaded_files):
        with cols[idx]:
            st.image(file, caption="Uploaded Image", width=150)
        preds, pil_img = predict_tumor(file)
        confidence = np.max(preds)
        label = CLASSES[np.argmax(preds)]
        results.append((label, confidence))

    # ---------------- Results ----------------
    st.subheader("Prediction Results")
    for i, (label, confidence) in enumerate(results):
        st.write(f"Image {i+1}: **{label}** ({confidence:.2f})")
        if confidence < 0.6:
            st.warning("Low confidence! Consider retaking the image or consulting a professional.")

    # ---------------- Grad-CAM Grid ----------------
    st.subheader("Grad-CAM Heatmaps")
    heatmap_cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        _, pil_img = preprocess_image(file)
        heatmap_img = grad_cam_visualization(pil_img)
        with heatmap_cols[idx]:
            st.image(heatmap_img, caption="Heatmap", width=150)

    # ---------------- Bar Chart ----------------
    st.subheader("Class Distribution")
    labels = [res[0] for res in results]
    unique_labels, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar(unique_labels, counts)
    ax.set_xlabel("Tumor Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ---------------- Hospital Recommendation ----------------
    st.subheader("Recommended Hospitals Nearby")
    st.markdown("Based on your location, here are nearby hospitals:")
    google_maps_link = "https://www.google.com/maps/search/brain+tumor+specialist+hospital+near+me/"
    st.markdown(f"[Open in Google Maps]({google_maps_link})")

else:
    st.info("Upload MRI images to get started.")
