import streamlit as st
import tensorflow as tf
import numpy as np
import requests

from io import BytesIO
from classes import sports_class
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("best_model.h5")

def resize_image(image, output_size):
    img_resized = image.resize(output_size)
    return img_resized
  
st.write("CPE019 - Final Project Model Deployment by Joseph Bryan M. Ferrer & John Glen Paz")
st.header("Sports Image Classification")
st.write("A deep learning model that uses EfficientNetB0 which is a convolutional neural network (CNN) architecture that predicts 100 classes of different sports.")

image_upload = st.file_uploader("Please upload an image depicting a sport in action.", type=["jpeg", "png"])
resized_image = None
sample_img_choice = st.button("Use Sample Image")

if image_upload is not None:
    img = Image.open(image_upload)
    st.image(img, caption="Uploaded Image")
    resized_image = resize_image(img, (224, 224))
else:
    image = Image.open("billiards.jpg")
    st.image(image, caption="Image", use_column_width=True)

    sample_image_path = "https://github.com/qjbferrer/CPE-019-Sports-Classification-Identifier/blob/main/images/billiards.jpg"
    sample_img = Image.open(sample_image_path)
    st.image(sample_img, caption="Here is a sample image")
    
if resized_image is not None:
    normalized_image = np.expand_dims(resized_image, axis=0)
    detections = model.predict(normalized_image)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = sports_class[class_index]  # Assuming class_names is defined somewhere
    st.success("Predicted sport: {}".format(sport_name))



if sample_img_choice:
    image = Image.open("test_cricket.jpg")
    st.image(image, caption="Image", use_column_width=True)
    label = predict_label(image, model)
    st.markdown(
        f"<h2 style='text-align: center;'>{label}</h2>",
        unsafe_allow_html=True,
    )
