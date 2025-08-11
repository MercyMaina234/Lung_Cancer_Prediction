import os
import streamlit as st
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

st.title("🫁 Lung Cancer Detection from CT Scans")
st.write("Upload a CT scan to detect signs of lung cancer using a pre-trained model.")

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(file_path, caption="Uploaded CT Scan", use_container_width=True)

    # Inference
    with st.spinner("Analyzing CT scan..."):
        result = CLIENT.infer(file_path, model_id="lung-cancer-oh7sx/1")

    # Extract predictions
    if "predictions" in result and len(result["predictions"]) > 0:
        pred = result["predictions"][0]  # first prediction
        label = pred["class"]  # class name from model
        confidence = pred["confidence"] * 100  # convert to %

        if label.lower() in ["cancer", "lung cancer", "positive"]:
            st.error(f"🚨 **Prediction: Lung Cancer Detected** ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ **Prediction: No Lung Cancer Detected** ({confidence:.2f}% confidence)")
    else:
        st.warning("No prediction returned. Please try another image.")

    # Optional: show raw output
    with st.expander("Raw Model Output"):
        st.json(result)
