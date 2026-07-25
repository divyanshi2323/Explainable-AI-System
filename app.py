import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Create SHAP explainer
explainer = shap.Explainer(model)

# UI Title
st.title("Explainable AI System for Heart Disease Prediction 🫀")
st.write("Predict heart disease risk using a Random Forest model with SHAP-based explainability.")

st.subheader("Enter Patient Details")

# Input fields
age = st.slider("Age", 20, 80, 40)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = True)", [0, 1])
restecg = st.slider("Rest ECG (0-2)", 0, 2, 1)
thalach = st.slider("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes)", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.slider("Slope (0-2)", 0, 2, 1)
ca = st.slider("Number of vessels (0-4)", 0, 4, 0)
thal = st.slider("Thal (0-3)", 0, 3, 1)

# Prediction button
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=[
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ])

    # Prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease ⚠️")
    else:
        st.success("Low Risk of Heart Disease ✅")

    # =========================
    # SHAP EXPLANATION SECTION
    # =========================
    st.subheader("📊 SHAP Explanation")

    shap_values = explainer(input_data)

    # Plot (handles multi-output safely)
    fig, ax = plt.subplots()
    try:
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
    except:
        shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)

    # =========================
    # TEXT EXPLANATION SECTION
    # =========================
    st.subheader("🔍 Key Factors Affecting the Prediction")

    values = shap_values.values

    # Handle multi-class
    if len(values.shape) == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    feature_names = input_data.columns

    feature_labels = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Rest ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Oldpeak",
    "slope": "Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia"
}

    # Create importance table
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    })

    importance_df["AbsImpact"] = abs(importance_df["Impact"])
    importance_df = importance_df.sort_values(by="AbsImpact", ascending=False)

    top_features = importance_df.head(5)

    # Display explanation
    for _, row in top_features.iterrows():
       feature = feature_labels[row["Feature"]]

       if row["Impact"] > 0:
          st.write(f"🔴 {feature} increased the risk")
       else:
          st.write(f"🔵 {feature} decreased the risk")

