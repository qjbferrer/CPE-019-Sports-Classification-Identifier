import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from classes import sports_class
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the trained model
model = load_model("./best_model.h5")


def resize_image(image, output_size):
    img_resized = image.resize(output_size)
    return img_resized


st.header("Upload a Image")

option = st.radio("Choose Image Input Method", ("Upload Image", "Provide URL"))

resized_image = None


if option == "Upload Image":
    image_upload = st.file_uploader("Upload An Image", type=["jpeg", "png"])
    if image_upload is not None:
        img = Image.open(image_upload)
        st.image(img, caption="Real Image")
        resized_image = resize_image(img, (224, 224))
elif option == "Provide URL":
    image_url = st.text_input("Enter Image URL")
    btn=st.button("Predict Image")
    if btn:
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Real Image")
                resized_image = resize_image(img, (224, 224))
            except Exception as e:
                st.error("Error downloading image from URL:", e)
                resized_image = None

if resized_image is not None:
    normalized_image_with_batch = np.expand_dims(resized_image, axis=0)
    detections = model.predict(normalized_image_with_batch)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = class_names[class_index]
    st.success("Predicted sport: {}".format(sport_name))
