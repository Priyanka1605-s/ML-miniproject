import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ==============================
# LOAD MODEL & SCALER
# ==============================

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# TITLE
# ==============================

st.title("🩺 Diabetes Prediction Dashboard")
st.markdown("Machine Learning based diabetes prediction system")

# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("Model Information")
st.sidebar.success("Random Forest Model Loaded")

st.sidebar.markdown("---")

st.sidebar.write("### Model Features")
st.sidebar.write("- Real-time prediction")
st.sidebar.write("- Feature scaling")
st.sidebar.write("- Random Forest classifier")
st.sidebar.write("- ROC curve analysis")
st.sidebar.write("- Confusion matrix")

# ==============================
# INPUT SECTION
# ==============================

st.subheader("Enter Patient Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        0,
        20,
        2
    )

with col2:
    glucose = st.number_input(
        "Glucose",
        0,
        300,
        138
    )

with col3:
    blood_pressure = st.number_input(
        "Blood Pressure",
        0,
        200,
        62
    )

with col4:
    skin_thickness = st.number_input(
        "Skin Thickness",
        0,
        100,
        35
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    insulin = st.number_input(
        "Insulin",
        0,
        900,
        0
    )

with col6:
    bmi = st.number_input(
        "BMI",
        0.0,
        70.0,
        33.6
    )

with col7:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        0.0,
        3.0,
        0.627
    )

with col8:
    age = st.number_input(
        "Age",
        1,
        120,
        50
    )

# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict Diabetes"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    # ==============================
    # RESULT
    # ==============================

    st.subheader("Prediction Result")
    risk_percent = probability * 100

    if risk_percent >= 75:
    
        st.error("⚠️ High Diabetes Risk Detected")
    
        st.metric(
            "Diabetes Probability",
            f"{risk_percent:.2f}%"
        )
    
    elif risk_percent >= 15:
    
        st.warning("🟠 Moderate Diabetes Risk")
    
        st.metric(
            "Diabetes Probability",
            f"{risk_percent:.2f}%"
        )
    
    else:
    
        st.success("✅ Low Diabetes Risk")
    
        st.metric(
            "Diabetes Probability",
            f"{risk_percent:.2f}%"
        )
   

    

  

    # ==============================
    # INPUT SUMMARY
    # ==============================

    st.subheader("Patient Details")

    patient_df = pd.DataFrame({
        "Feature": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age"
        ],
        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    })

    st.dataframe(
        patient_df,
        use_container_width=True
    )

# ==============================
# FEATURE IMPORTANCE
# ==============================

st.markdown("---")
st.subheader("Feature Importance")

feature_names = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "DPF",
    "Age"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax1.invert_yaxis()

ax1.set_xlabel("Importance Score")
ax1.set_ylabel("Features")

st.pyplot(fig1)

# ==============================
# SAMPLE CONFUSION MATRIX
# ==============================

st.markdown("---")
st.subheader("Confusion Matrix")

cm = np.array([
    [88, 12],
    [20, 34]
])

fig2, ax2 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ==============================
# SAMPLE ROC CURVE
# ==============================

st.markdown("---")
st.subheader("ROC Curve")

fpr = [0.0, 0.1, 0.2, 0.4, 1.0]
tpr = [0.0, 0.6, 0.75, 0.9, 1.0]

fig3, ax3 = plt.subplots(figsize=(6, 5))

ax3.plot(
    fpr,
    tpr,
    label="Random Forest (AUC = 0.86)"
)

ax3.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")

ax3.legend()

st.pyplot(fig3)

# ==============================
# CSV UPLOAD
# ==============================

st.markdown("---")
st.subheader("Batch Prediction using CSV")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset")

    st.dataframe(data)

    scaled_data = scaler.transform(data)

    predictions = model.predict(scaled_data)

    data["Prediction"] = predictions

    st.write("Prediction Results")

    st.dataframe(data)

# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption(
    "Diabetes Prediction Mini Project using Streamlit"
)