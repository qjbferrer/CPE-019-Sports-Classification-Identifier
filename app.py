import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image

# Load class names from your saved training (or hardcode sports_class)
from classes import sports_class  
NUM_CLASSES = len(sports_class)  # matches your dataset

# ========================
# Rebuild trained model architecture
# ========================
def build_model(num_classes):
    base_model = EfficientNetB0(
        weights=None,  # don't load imagenet here since weights will come from .h5
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    inputs = keras.Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Build model and load weights
model = build_model(NUM_CLASSES)
model.load_weights("image_classifier_best.h5")  # load only weights

# ========================
# Preprocessing function
# ========================
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB")                           # force 3 channels
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)        # (1,224,224,3)
    return img_array

# ========================
# Streamlit UI
# ========================
st.write("CPE019 - Final Project Model Deployment by Joseph Bryan M. Ferrer & John Glen Paz")
st.header("Sports Image Classification")
st.write("A deep learning model that uses EfficientNetB0 to classify images into 100 different sports.")

image_upload = st.file_uploader("Upload a sports image", type=["jpeg", "jpg", "png"])
image_to_predict = None

if image_upload is not None:
    img = Image.open(image_upload).convert("RGB")
    st.image(img, caption="Uploaded Image")
    image_to_predict = img
else:
    st.write("Or try with the sample image:")
    if st.button("Use Sample Image"):
        img = Image.open("images/billiards.jpg").convert("RGB")
        st.image(img, caption="Sample Image", use_column_width=True)
        image_to_predict = img

# ========================
# Prediction
# ========================
if image_to_predict is not None:
    processed_img = preprocess_image(image_to_predict)
    detections = model.predict(processed_img)
    class_index = np.argmax(detections, axis=1)[0]
    sport_name = sports_class[class_index]
    st.success(f"Predicted sport: {sport_name}")
