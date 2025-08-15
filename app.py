import streamlit as st
import tensorflow as tf
import numpy as np
from classes import sports_class
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("best_model.h5")

def resize_image(image, output_size):
    img_resized = image.resize(output_size)
    return img_resized

st.write("CPE019 - Final Project Model Deployment by Joseph Bryan M. Ferrer & John Glen Paz")
st.header("Sports Image Classification")
st.write("A deep learning model that uses EfficientNetB0, a convolutional neural network (CNN) architecture, to predict 100 classes of different sports.")

image_upload = st.file_uploader("Please upload an image depicting a sport in action.", type=["jpeg", "png", "jpg"])
resized_image = None

if image_upload is not None:
    img = Image.open(image_upload).convert("RGB")  # ✅ Ensure 3 channels
    st.image(img, caption="Uploaded Image")
    resized_image = resize_image(img, (224, 224))
else:
    st.write("If you prefer not to upload an image, you have the option to use the provided sample image below.")
    if st.button("Sample Image"):
        image = Image.open("images/billiards.jpg").convert("RGB")  # ✅ Ensure 3 channels
        st.image(image, caption="Sample Image", use_column_width=True)
        resized_image = resize_image(image, (224, 224))
    
if resized_image is not None:
    normalized_image = np.array(resized_image) / 255.0  # ✅ Normalize to match training preprocessing
    normalized_image = np.expand_dims(normalized_image, axis=0)  # Shape: (1, 224, 224, 3)

    detections = model.predict(normalized_image)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = sports_class[class_index]

    st.success(f"Predicted sport: {sport_name}")
