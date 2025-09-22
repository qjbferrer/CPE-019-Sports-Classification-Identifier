import streamlit as st
import tensorflow as tf
import numpy as np
from classes import sports_class
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image

# ========================
# Rebuild model architecture
# ========================
NUM_CLASSES = 100  # adjust to your dataset

base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Load weights instead of full model
model.load_weights("best_model.h5")

# ========================
# Preprocessing function
# ========================
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")                           # force RGB
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)        # (1, H, W, 3)
    return img_array

# ========================
# Streamlit UI
# ========================
st.write("CPE019 - Final Project Model Deployment by Joseph Bryan M. Ferrer & John Glen Paz")
st.header("Sports Image Classification")
st.write("A deep learning model that uses EfficientNetB0 trained on RGB images to predict 100 classes of different sports.")

image_upload = st.file_uploader("Please upload an image depicting a sport in action.", type=["jpeg", "png"])
image_to_predict = None

if image_upload is not None:
    img = Image.open(image_upload).convert("RGB")
    st.image(img, caption="Uploaded Image")
    image_to_predict = img
else:
    st.write("If you prefer not to upload an image, you have the option to use the provided sample image below.")
    if st.button("Sample Image"):
        img = Image.open("images/billiards.jpg").convert("RGB")
        st.image(img, caption="Sample Image", use_column_width=True)
        image_to_predict = img

# ========================
# Prediction
# ========================
if image_to_predict is not None:
    processed_img = preprocess_image(image_to_predict, (224, 224))
    detections = model.predict(processed_img)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = sports_class[class_index]
    st.success(f"Predicted sport: {sport_name}")
