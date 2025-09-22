import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pickle

# ---------------------------
# Load model and class names
# ---------------------------
@st.cache_resource
def load_model_and_classes():
    # Load the entire model (not just weights)
    model = keras.models.load_model("image_classifier_best.h5")

    # Load the class names saved during training
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)

    return model, class_names

model, class_names = load_model_and_classes()

# ---------------------------
# App UI
# ---------------------------
st.title("🏅 Sports Image Classifier")
st.write("Upload a sports image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0   # normalize like in training
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Show results
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Prediction:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Debug (optional)
    st.write("🔎 Raw probabilities:", predictions[0])
