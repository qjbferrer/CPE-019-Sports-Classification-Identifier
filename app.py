import streamlit as st
import tensorflow as tf
import numpy as np
from classes import sports_class
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("best_model.h5")

def resize_image(image, output_size):
    # Convert to grayscale ("L") before resizing
    img_resized = image.convert("L")
    img_resized = img_resized.resize(output_size)
    return img_resized

st.write("CPE019 - Final Project Model Deployment by Joseph Bryan M. Ferrer & John Glen Paz")
st.header("Sports Image Classification")
st.write("A deep learning model that uses EfficientNetB0 trained on grayscale images to predict 100 classes of different sports.")

# Upload or use sample image
image_upload = st.file_uploader("Please upload an image depicting a sport in action.", type=["jpeg", "png"])
resized_image = None

if image_upload is not None:
    img = Image.open(image_upload).convert("L")  # force grayscale
    st.image(img, caption="Uploaded Image")
    resized_image = resize_image(img, (224, 224))
else:
    st.write("If you prefer not to upload an image, you have the option to use the provided sample image below.")
    if st.button("Sample Image"):
        image = Image.open("images/billiards.jpg").convert("L")  # force grayscale
        st.image(image, caption="Sample Image", use_column_width=True)
        resized_image = resize_image(image, (224, 224))

# Prediction
if resized_image is not None:
    # Convert to array and normalize
    resized_image = np.array(resized_image).astype("float32") / 255.0
    # Add channel dimension (H, W, 1)
    resized_image = np.expand_dims(resized_image, axis=-1)
    # Add batch dimension (1, H, W, 1)
    normalized_image = np.expand_dims(resized_image, axis=0)

    detections = model.predict(normalized_image)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = sports_class[class_index]
    st.success(f"Predicted sport: {sport_name}")
