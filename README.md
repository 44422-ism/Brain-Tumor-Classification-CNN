# 🧠 Brain Tumor Detection App

## 📌 Project Overview

This project is a **machine learning-based prototype** that detects the presence and type of brain tumors from MRI images. The model is trained on a limited dataset using Google Colab and deployed as an interactive **Streamlit web application**.

The app allows users to:

* Upload MRI scans.
* Detect tumors (if present).
* View classification results along with a **confidence score**.
* Access a **hospital recommendation system** (basic version).

---

## ⚙️ Tech Stack

### **Frontend (Streamlit Web App)**

* **Streamlit** → User-friendly web interface.
* **Python** → For integrating the ML model into the app.

### **Backend (ML Model & Training)**

* **TensorFlow / Keras** → For CNN-based tumor classification.
* **Google Colab (.ipynb)** → Model training environment.
* **GitHub** → Code version control and collaboration.

---

## 📂 Directory Structure

```
app.py                  # Main Streamlit application  
/model/                 # Trained tumor detection model (.h5 file)  
/dataset/               # Dataset (organized into class folders)  
notebooks/training.ipynb# Colab notebook for training  
requirements.txt        # Python dependencies  
```

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🔬 Limitations (Current Version)

* Model is trained on a **limited dataset** → may not generalize well.
* Predictions may be **biased** and lack reliability in real-world cases.
* Requires **retraining** for better accuracy.
* No real-time hospital integration yet.
* Confidence score reflects probability but not **clinical reliability**.

---

## 🌱 Future Scope

* Integration with **smart hospital recommendation system** (API-based).
* Implementation of **incremental/reinforcement learning** for adaptability.
* Larger, diverse datasets for **improved generalization**.
* Enhanced interpretability using **heatmaps or Grad-CAM visualizations**.
* Compliance with **medical data ethics and privacy**.

---

## 📦 Software Requirements

* Python 3.8+
* TensorFlow / Keras
* Streamlit
* NumPy, Pandas, Matplotlib
* scikit-learn

---

## 🏥 Business & Impact

This prototype demonstrates how **AI in healthcare** can help in:

* Early detection of brain tumors.
* Supporting radiologists with **decision assistance**.
* Laying the foundation for **accessible and scalable diagnostic tools**.
  
---

✨ *This is a prototype project created for academic/research purposes, not for clinical diagnosis.*
