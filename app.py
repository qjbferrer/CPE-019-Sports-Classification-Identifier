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
  
st.write("CPE019 - Final Project Model Deployment)
st.header("Sports Image Classification")

image_upload = st.file_uploader("Upload An Image", type=["jpeg", "png"])
resized_image = None
