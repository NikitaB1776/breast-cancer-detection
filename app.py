import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    with open("breast_cancer_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==============================
# Feature Names
# ==============================
FEATURES = [
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses"
]

# ==============================
# Title
# ==============================
st.title("🩺 Breast Cancer Detection")
st.write("Machine Learning based early cancer prediction")
st.write("---")

# ==============================
# Sidebar: Input Method
# ==============================
option = st.sidebar.radio(
    "Input Method",
    ("Manual Input", "Upload CSV File")
)

# ==============================
# Manual Input
# ==============================
if option == "Manual Input":
    st.subheader("Enter Cell Features (1–10)")

    user_input = []
    for feature in FEATURES:
        value = st.number_input(feature, min_value=1, max_value=10, value=1)
        user_input.append(value)

    input_data = np.array(user_input).reshape(1, -1)

    if st.button("🔬 Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 4:
            st.error(f"⚠️ Malignant Tumor Detected (Confidence: {max(probability)*100:.2f}%)")
        else:
            st.success(f"✅ Benign Tumor Detected (Confidence: {max(probability)*100:.2f}%)")

# ==============================
# CSV Upload
# ==============================
else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV with 9 features", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        if list(data.columns) != FEATURES:
            st.error("⚠️ CSV must contain the 9 training features in this exact order.")
        else:
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)

            results = data.copy()
            results["Prediction"] = ["Malignant" if p == 4 else "Benign" for p in predictions]
            results["Confidence (%)"] = [round(max(prob)*100, 2) for prob in probabilities]

            st.write("🧾 Prediction Results")
            st.dataframe(results)

# ==============================
# Disclaimer
# ==============================
st.write("---")
st.markdown("""
⚠️ **Medical Disclaimer**  
This application is for educational purposes only.  
Not a substitute for professional medical advice.
""")
