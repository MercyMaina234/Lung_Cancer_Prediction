import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Lung Cancer Predictor", layout="centered")

# App title and description
st.title("🫁 Lung Cancer Risk Predictor")
st.markdown("""
Welcome to the Lung Cancer Predictor App.  
Please fill in the details below to predict your risk of lung cancer.
""")

# Sidebar - About
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.header("About")
    st.write("""
    This app predicts lung cancer risk based on health-related attributes.  
    Model: Random Forest  
    Data: Kaggle Lung Cancer Dataset  
    """)

# Collect user input
def get_user_input():
    user_data = {}
    for col in columns:
        label = col.replace('_', ' ').title()

        if col.lower() == 'gender':
            gender = st.selectbox('What is your gender?', ['Male', 'Female'], key=col)
            user_data[col] = 1 if gender == 'Male' else 0

        elif 'age' in col.lower():
            user_data[col] = st.number_input(f"{label}:", min_value=10, max_value=120, value=30, key=col)
        elif col.lower() == 'smoking':
            choice = st.selectbox('Do you smoke?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0
        elif col.lower() == 'yellow fingers':
            choice = st.selectbox('Do you have yellow fingers?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0
        elif col.lower() == 'chronic disease':
            choice = st.selectbox('Do you have any chronic diseases?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0


        elif col.lower() == 'alcohol consuming':
            choice = st.selectbox('Do drink alcohol?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0
        elif col.lower() == 'chest pain':
            choice = st.selectbox('Do experience chest pain?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0
        elif col.lower() == 'coughing':
            choice = st.selectbox('Do you cough?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0


        elif col.lower() == 'anxiety':
            choice = st.selectbox('Do you struggle with anxiety?', ['Yes', 'No'], key=col)
            user_data[col] = 1 if choice == 'Yes' else 0

        else:
            user_data[col] = st.slider(f"{col.replace('_', ' ').title()}:", 0, 10, 5)







    return pd.DataFrame([user_data])



input_df = get_user_input()

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")

    st.write(f"📊 Probability: `{prob:.2%}`")

    # Probability donut chart
    fig, ax = plt.subplots()
    ax.pie([prob, 1 - prob], labels=["Cancer Risk", "No Risk"], autopct='%1.1f%%', startangle=90, colors=["red", "green"])
    ax.axis('equal')
    st.pyplot(fig)

# Optional: Show feature importance
st.markdown("---")
if st.checkbox("🔎 Show Feature Importance (from model)"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)
    
st.markdown("""
---
⚠ **Disclaimer:**  
This prediction tool is for educational and informational purposes only.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult with a qualified healthcare provider regarding any medical concerns.
""")