# app.py (updated)
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import pandas as pd
import urllib.parse
import math

# -------------------------------
# Paths / constants
# -------------------------------
MRI_MODEL_PATH = "multi_class_mri_detector.tflite"   # must exist in folder
TUMOR_MODEL_PATH = "tumor_model.tflite"             # your new model
MRI_CLASSES = ["Brain MRI", "Other MRI", "Not MRI"]
TUMOR_CLASSES = ['No Tumor', 'Glioma', 'Pituitary', 'Meningioma']

# thresholds (tune these after testing)
MRI_CONF_THRESHOLD = 0.80        # only run tumor model if MRI detector confidence >= this
PRESENCE_MAX_THRESH = 0.40      # for tumor presence: max tumor-class score across patches must exceed this
CLASS_MEAN_THRESH = 0.25        # mean prob threshold for chosen class across patches
PATCH_SIZE_DEFAULT = 224
PATCH_STRIDE = 112

CONFIDENCE_THRESHOLD = 0.5  # app-level uncertainty warning

# simple hospital table
HOSPITALS = {
    "New Delhi, India": ["AIIMS", "Max Super Speciality Hospital"],
    "Mumbai, India": ["Tata Memorial Hospital", "Lilavati Hospital"],
    "Bangalore, India": ["Manipal Hospital", "Apollo Hospital"],
    # ... (keep your existing entries)
}

# -------------------------------
# Load TFLite interpreters (cached)
# -------------------------------
@st.cache_resource
def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

mri_interpreter = load_interpreter(MRI_MODEL_PATH)
tumor_interpreter = load_interpreter(TUMOR_MODEL_PATH)

# -------------------------------
# Helper: model I/O utilities for quantized models
# -------------------------------
def get_input_shape(interpreter):
    d = interpreter.get_input_details()[0]
    shape = d['shape']  # e.g. [1,224,224,3]
    return tuple(shape)

def preprocess_for_interpreter(img: Image.Image, interpreter):
    """
    Resize/normalize and quantize/scale as required by interpreter.
    Returns array ready to set_tensor.
    """
    input_det = interpreter.get_input_details()[0]
    shape = input_det['shape']
    h, w = int(shape[1]), int(shape[2])
    dtype = input_det['dtype']

    # ensure RGB if model expects 3 channels
    channels = shape[3] if len(shape) > 3 else 1
    if channels == 3 and img.mode != "RGB":
        img = img.convert("RGB")
    if channels == 1 and img.mode != "L":
        img = img.convert("L")

    img_resized = img.resize((w, h))
    arr = np.array(img_resized).astype('float32') / 255.0  # normalized 0-1

    # if channels mismatch (rare), broadcast
    if arr.ndim == 2 and channels == 3:
        arr = np.stack([arr]*3, axis=-1)

    arr = np.expand_dims(arr, axis=0)  # add batch

    # quantization handling
    scale, zero_point = input_det.get('quantization', (0.0, 0))
    if dtype == np.uint8 and scale != 0:
        # convert to quantized representation
        arr_q = (arr / scale).round() + zero_point
        arr_q = np.clip(arr_q, 0, 255).astype(np.uint8)
        return arr_q
    else:
        # float model
        if dtype == np.float32:
            return arr.astype(np.float32)
        else:
            return arr.astype(dtype)

def run_tflite(interpreter, input_arr):
    """Handles quantization on input and dequantization for output if needed."""
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_arr)
    interpreter.invoke()
    out_details = interpreter.get_output_details()[0]
    raw_output = interpreter.get_tensor(out_details['index'])

    # handle quantized output: convert to float if needed
    scale, zero_point = out_details.get('quantization', (0.0, 0))
    if out_details['dtype'] == np.uint8 and scale != 0:
        return ((raw_output.astype(np.float32) - zero_point) * scale)[0]
    else:
        return raw_output[0]

# -------------------------------
# Patching utilities
# -------------------------------
def extract_patches(img: Image.Image, patch_size=PATCH_SIZE_DEFAULT, stride=PATCH_STRIDE):
    w, h = img.size
    patches = []
    positions = []
    # pad so that edges are covered
    if w < patch_size or h < patch_size:
        # single centered patch scaled
        patch = img.resize((patch_size, patch_size))
        return [patch], [(0,0)]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x,y))
    # ensure coverage right/bottom edges
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x,y))
    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x,y))
    # corner
    if (w - patch_size) % stride != 0 and (h - patch_size) % stride != 0:
        patch = img.crop((w - patch_size, h - patch_size, w, h))
        patches.append(patch)
        positions.append((w - patch_size, h - patch_size))
    return patches, positions

# Robust aggregation strategy
def aggregate_patch_predictions(patch_preds):
    """
    patch_preds: array shape (num_patches, num_classes) of probabilities
    returns: mean_probs, max_per_class, final_label, final_confidence
    """
    mean_probs = patch_preds.mean(axis=0)
    max_probs = patch_preds.max(axis=0)
    chosen = int(np.argmax(mean_probs))
    chosen_mean = float(mean_probs[chosen])
    chosen_max = float(max_probs[chosen])
    return mean_probs, max_probs, chosen, chosen_mean, chosen_max

# heatmap - only draw overlay where patch confidence for chosen class is high
def generate_patch_heatmap(img: Image.Image, positions, chosen_class_conf_per_patch, patch_size=PATCH_SIZE_DEFAULT, min_conf=0.4):
    heat = Image.new("RGBA", img.size, (0,0,0,0))
    for (x,y), conf in zip(positions, chosen_class_conf_per_patch):
        if conf >= min_conf:
            intensity = int(min(1.0, conf) * 180)  # alpha
            overlay = Image.new("RGBA", (patch_size, patch_size), (255,0,0,intensity))
            heat.paste(overlay, (x,y), overlay)
    return Image.alpha_composite(img.convert("RGBA"), heat)

# -------------------------------
# App UI & logic
# -------------------------------
st.set_page_config(page_title="Brain Tumor Classifier (Robust)", layout="wide")
st.title("🧠 Brain Tumor Classification + MRI pre-filter")

st.markdown("""
**Disclaimer:** This tool is for educational purposes only. Predictions are not 100% accurate. Always consult a medical professional.
""")

# sidebar
st.sidebar.header("Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body {background-color:#111;color:#eee}</style>", unsafe_allow_html=True)

st.sidebar.header("Hospitals")
location_input = st.sidebar.text_input("Enter your city and country (e.g., New Delhi, India)")
if location_input:
    top_hosp = HOSPITALS.get(location_input, ["No predefined hospitals for this location"])
    for h in top_hosp:
        link = f"https://www.google.com/maps/search/{urllib.parse.quote(h + ', ' + location_input)}"
        st.sidebar.markdown(f"- [{h}]({link})", unsafe_allow_html=True)

st.sidebar.header("Scan history")
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

uploaded_files = st.file_uploader("Upload MRI images (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=True)

results = []
if uploaded_files:
    st.subheader("Predictions & heatmaps")
    # create columns grid (up to 4 per row)
    cols_count = min(4, len(uploaded_files))
    cols = st.columns(cols_count)

    for idx, f in enumerate(uploaded_files):
        # load image
        img = Image.open(f)
        # --- MRI pre-filter ---
        mri_arr = preprocess_for_interpreter(img, mri_interpreter)
        mri_out = run_tflite(mri_interpreter, mri_arr)
        # mri_out is array of probs over MRI_CLASSES
        mri_label = MRI_CLASSES[int(np.argmax(mri_out))]
        mri_conf = float(np.max(mri_out))

        if mri_label != "Brain MRI" or mri_conf < MRI_CONF_THRESHOLD:
            # skip tumor classification, mark appropriately
            label = f"Not Brain MRI ({mri_label})"
            conf_score = mri_conf
            summary_df = pd.DataFrame({
                'Class': TUMOR_CLASSES,
                'Probability': [0.0]*len(TUMOR_CLASSES)
            })
            info = f"Skipped tumor classifier because MRI detector returned: {mri_label} (conf {mri_conf:.2f})."
            # show thumbnail and message
            cols[idx % cols_count].image(img, caption=f"{f.name} — {label} (MRI conf {mri_conf:.2f})", use_column_width=True)
            st.warning(f"{f.name}: MRI detector says {mri_label} (conf {mri_conf:.2f}). Tumor classifier skipped.")
        else:
            # Run patch-based tumor classification
            # get tile size from tumor interp input
            t_in_shape = get_input_shape(tumor_interpreter)
            psize = t_in_shape[1] if len(t_in_shape) > 1 else PATCH_SIZE_DEFAULT
            patches, positions = extract_patches(img, patch_size=psize, stride=math.floor(psize/2))

            patch_preds = []
            for p in patches:
                arr = preprocess_for_interpreter(p, tumor_interpreter)
                out = run_tflite(tumor_interpreter, arr)  # per-patch probabilities
                # ensure length
                if out.shape[0] != len(TUMOR_CLASSES):
                    # pad/trim
                    tmp = np.zeros(len(TUMOR_CLASSES), dtype=np.float32)
                    for i in range(min(len(out), len(tmp))):
                        tmp[i] = out[i]
                    out = tmp
                patch_preds.append(out)
            patch_preds = np.array(patch_preds)  # (N, C)

            mean_probs, max_probs, chosen_idx, chosen_mean, chosen_max = aggregate_patch_predictions(patch_preds)

            # decide final label with conservative checks:
            # require that the chosen_mean exceed CLASS_MEAN_THRESH and chosen_max exceed PRESENCE_MAX_THRESH
            if chosen_mean < CLASS_MEAN_THRESH or chosen_max < PRESENCE_MAX_THRESH:
                # consider it No Tumor (or Uncertain)
                final_label = "No Tumor"
                final_conf = float(max_probs[TUMOR_CLASSES.index('No Tumor')] if 'No Tumor' in TUMOR_CLASSES else max_probs.max())
            else:
                final_label = TUMOR_CLASSES[chosen_idx]
                final_conf = float(chosen_max)  # use more optimistic patch-level confidence

            # build summary table
            summary_df = pd.DataFrame({'Class': TUMOR_CLASSES, 'Probability': [float(x) for x in mean_probs]}).sort_values(by='Probability', ascending=False)

            # generate a patch-based heatmap emphasizing patches where chosen class prob is high
            chosen_class_patch_conf = patch_preds[:, chosen_idx] if chosen_idx < patch_preds.shape[1] else np.zeros(len(patch_preds))
            heatmap_img = generate_patch_heatmap(img, positions, chosen_class_patch_conf, patch_size=psize, min_conf=0.35)

            # display
            cols[idx % cols_count].image(heatmap_img, caption=f"{f.name} — {final_label} (conf {final_conf:.2f})", use_column_width=True)

            # confidence warning
            if final_conf < CONFIDENCE_THRESHOLD:
                st.warning(f"{f.name}: Tumor classification uncertain ({final_conf:.2f}). Consider specialist review.")

            label = final_label
            conf_score = final_conf
            info_map = {
                "No Tumor": "No detectable brain tumor in the scan.",
                "Glioma": "Gliomas are tumors originating in glial cells. Can be low or high grade.",
                "Meningioma": "Meningiomas develop from the meninges, the brain's protective layers.",
                "Pituitary": "Pituitary tumors form in the pituitary gland affecting hormonal balance.",
            }
            info = info_map.get(label, "No info available.")

            # probability bar chart
            st.subheader(f"Prediction probabilities — {f.name}")
            st.bar_chart(summary_df.set_index('Class'))

        # record result
        results.append({
            "File Name": f.name,
            "MRI Label": mri_label,
            "MRI Conf": round(float(mri_conf), 2),
            "Tumor Type": label,
            "Tumor Confidence": round(float(conf_score), 2),
            "Info": info
        })

        # update session history
        st.session_state.scan_history.append(results[-1])

    # summary table
    st.subheader("📊 Summary")
    df = pd.DataFrame(results)
    st.dataframe(df)

else:
    st.info("Upload one or more MRI images. Pre-filter protects against non-brain images.")
