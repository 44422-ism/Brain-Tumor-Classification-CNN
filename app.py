import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# -------------------------------
# Constants
# -------------------------------
TUMOR_MODEL_PATH = "tumor_model.tflite"
TUMOR_CLASSES = ['No Tumor', 'Glioma', 'Pituitary', 'Meningioma']

# Hospital recommendations
HOSPITALS = {
    "New Delhi, India": ["AIIMS", "Max Super Speciality Hospital"],
    "Mumbai, India": ["Tata Memorial Hospital", "Lilavati Hospital"],
    "Bangalore, India": ["Manipal Hospital", "Apollo Hospital"],
    "Chennai, India": ["Apollo Specialty Hospital", "Fortis Malar Hospital"],
    "Hyderabad, India": ["Yashoda Hospital", "Care Hospital"],
    "London, UK": ["The Royal London Hospital", "King's College Hospital"],
    "Paris, France": ["Pitié-Salpêtrière Hospital", "Cochin Hospital"],
    "Berlin, Germany": ["Charité – Universitätsmedizin Berlin", "Helios Klinikum Berlin-Buch"],
    "Rome, Italy": ["Policlinico Gemelli", "Ospedale San Camillo"],
    "Madrid, Spain": ["Hospital Universitario La Paz", "Hospital 12 de Octubre"],
    "New York, USA": ["Memorial Sloan Kettering", "NYU Langone Health"],
    "Los Angeles, USA": ["UCLA Medical Center", "Cedars-Sinai Medical Center"],
    "Chicago, USA": ["Northwestern Memorial Hospital", "Rush University Medical Center"],
    "Toronto, Canada": ["Toronto General Hospital", "Sunnybrook Health Sciences Centre"],
    "Sydney, Australia": ["Royal Prince Alfred Hospital", "Sydney Adventist Hospital"]
}

# -------------------------------
# Load TFLite Model
# -------------------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

tumor_interpreter = load_tflite_model(TUMOR_MODEL_PATH)

# -------------------------------
# Functions
# -------------------------------
def preprocess_image(img, input_shape=(224,224,3)):
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to model input size
    img = img.resize((input_shape[1], input_shape[0]))
    # Normalize and expand dims
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tumor(uploaded_file):
    img = Image.open(uploaded_file)
    # Use correct input shape
    img_array = preprocess_image(img, input_shape=(224,224,3))

    input_details = tumor_interpreter.get_input_details()
    output_details = tumor_interpreter.get_output_details()

    tumor_interpreter.set_tensor(input_details[0]['index'], img_array)
    tumor_interpreter.invoke()
    preds = tumor_interpreter.get_tensor(output_details[0]['index'])[0]

    # Fix if output length differs
    if len(preds) != len(TUMOR_CLASSES):
        preds_fixed = np.zeros(len(TUMOR_CLASSES))
        for i in range(min(len(preds), len(TUMOR_CLASSES))):
            preds_fixed[i] = preds[i]
        preds = preds_fixed

    label_index = int(np.argmax(preds))
    label = TUMOR_CLASSES[label_index]
    conf_score = float(preds[label_index])

    # Summary table
    summary_df = pd.DataFrame({
        'Class': TUMOR_CLASSES,
        'Probability': [float(p) for p in preds]
    }).sort_values(by='Probability', ascending=False)

    # Tumor descriptive info
    tumor_info_dict = {
        "No Tumor": "No detectable brain tumor in the scan.",
        "Glioma": "Gliomas are tumors originating in glial cells. Can be low or high grade.",
        "Meningioma": "Meningiomas develop from the meninges, the brain's protective layers.",
        "Pituitary": "Pituitary tumors form in the pituitary gland affecting hormonal balance.",
        "Uncertain": "Prediction is uncertain. Please consult a specialist."
    }
    info = tumor_info_dict.get(label, "No info available.")

    return label, conf_score, summary_df, info

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor Classification")

st.markdown("""
**Disclaimer:** These Predictions are based on a trained model and are not 100% accurate. Always consult a medical professional.
""")

uploaded_file = st.file_uploader("Upload MRI image (jpg/png)", type=["jpg","jpeg","png"])

# Sidebar widgets
st.sidebar.header("📊 Scan History & Info")
scan_history = st.sidebar.empty()
tumor_info_widget = st.sidebar.empty()
st.sidebar.header("🏥 Hospital Recommendations")
location_input = st.sidebar.text_input("Enter your city and country (e.g., New Delhi, India)")

# Display top hospitals
if location_input:
    top_hospitals = HOSPITALS.get(location_input, ["Location not found. Choose from predefined locations."])
    st.sidebar.write("Top Hospitals:")
    for h in top_hospitals:
        st.sidebar.write(f"- {h}")

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file:
    try:
        label, conf_score, summary, info = predict_tumor(uploaded_file)

        st.write(f"**Predicted Class:** {label}")
        st.write(f"**Confidence Score:** {conf_score:.2f}")

        # Display summary table
        st.subheader("Prediction Summary")
        st.dataframe(summary)

        # Update sidebar widgets
        scan_history.markdown(f"**Last Scan:** {uploaded_file.name} - {label} ({conf_score:.2f})")
        tumor_info_widget.markdown(f"**Tumor Info:** {info}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
