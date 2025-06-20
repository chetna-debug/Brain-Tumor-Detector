import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("model/brain_tumor_model.h5")

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an MRI image and the model will predict if a brain tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    
    # Preprocess image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 100, 100, 1)

    # Make prediction
    prediction = model.predict(image_array)
    label = np.argmax(prediction)

    # Display result
    if label == 1:
        st.error("🔴 Brain Tumor Detected")
    else:
        st.success("🟢 No Brain Tumor Detected")
