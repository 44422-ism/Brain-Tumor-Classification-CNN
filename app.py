import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import pandas as pd
import urllib.parse

# -------------------------------
# Constants
# -------------------------------
TUMOR_MODEL_PATH = "tumor_model.tflite"
TUMOR_CLASSES = ['No Tumor', 'Glioma', 'Pituitary', 'Meningioma']
CONFIDENCE_THRESHOLD = 0.5  # Threshold for uncertainty warning

# Hospital base data (can be extended)
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
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((input_shape[1], input_shape[0]))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tumor(img):
    img_array = preprocess_image(img)
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

    summary_df = pd.DataFrame({
        'Class': TUMOR_CLASSES,
        'Probability': [float(p) for p in preds]
    }).sort_values(by='Probability', ascending=False)

    tumor_info_dict = {
        "No Tumor": "No detectable brain tumor in the scan.",
        "Glioma": "Gliomas are tumors originating in glial cells. Can be low or high grade.",
        "Meningioma": "Meningiomas develop from the meninges, the brain's protective layers.",
        "Pituitary": "Pituitary tumors form in the pituitary gland affecting hormonal balance.",
        "Uncertain": "Prediction is uncertain. Please consult a specialist."
    }
    info = tumor_info_dict.get(label, "No info available.")

    return label, conf_score, summary_df, info

def generate_heatmap(img, conf_score, patch_size=224):
    # Simple heat overlay proportional to confidence
    overlay = Image.new("RGBA", img.size, (255,0,0,int(conf_score*150)))
    return Image.alpha_composite(img.convert("RGBA"), overlay)

def get_google_maps_link(hospital_name, location):
    query = f"{hospital_name}, {location}"
    return f"https://www.google.com/maps/search/{urllib.parse.quote(query)}"

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor Classification")
st.markdown("""
**Disclaimer:** Predictions are based on a trained model and are not 100% accurate. Always consult a medical professional for cross examination.
""")

# Sidebar Widgets
st.sidebar.header("⚙ App Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("<style>body {background-color:#222;color:#eee}</style>", unsafe_allow_html=True)

st.sidebar.header("📊 Scan History & Info")
scan_history = st.sidebar.empty()
tumor_info_widget = st.sidebar.empty()
st.sidebar.header("🏥 Hospital Recommendations")
location_input = st.sidebar.text_input("Enter your city and country (e.g., New Delhi, India)")

# File uploader
uploaded_files = st.file_uploader("Upload MRI images", type=["jpg","jpeg","png"], accept_multiple_files=True)
results = []

# -------------------------------
# Prediction Loop
# -------------------------------
if uploaded_files:
    st.subheader("🔬 Predictions & Heatmaps")
    cols = st.columns(len(uploaded_files)) if len(uploaded_files) <= 4 else st.columns(4)
    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        label, conf_score, summary, info = predict_tumor(img)

        # Confidence warning
        if conf_score < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠ Prediction for {file.name} is uncertain (Confidence: {conf_score:.2f}). Consult a medical professional.")

        # Heatmap
        heatmap_img = generate_heatmap(img, conf_score)
        cols[i % len(cols)].image(heatmap_img, caption=f"{file.name} - {label}", use_column_width=True)

        # Bar chart of probabilities
        st.subheader(f"Prediction Probabilities for {file.name}")
        st.bar_chart(summary.set_index('Class'))

        # Record results
        results.append({
            "File Name": file.name,
            "Tumor Type": label,
            "Confidence": round(conf_score,2),
            "Info": info
        })

    # Summary table
    st.subheader("📊 Summary Table")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Sidebar update
    last_scan = results[-1]
    scan_history.markdown(f"**Last Scan:** {last_scan['File Name']} - {last_scan['Tumor Type']} ({last_scan['Confidence']:.2f})")
    tumor_info_widget.markdown(f"**Tumor Info:** {last_scan['Info']}")

    # Hospital recommendations with Google Maps links
    if location_input:
        st.sidebar.write("Top Hospitals:")
        top_hospitals = HOSPITALS.get(location_input, ["No predefined hospitals for this location"])
        for h in top_hospitals:
            link = get_google_maps_link(h, location_input)
            st.sidebar.markdown(f"- [{h}]({link})", unsafe_allow_html=True)

else:
    st.warning("Please upload one or more MRI images to get predictions.")
