import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests

# ---------------------------
# Load model and class names
# ---------------------------
@st.cache_resource
def load_model_and_classes():
    # Load the trained model
    model = keras.models.load_model("best_model.h5")

    # Fetch class names from GitHub (raw file, not HTML page)
    github_url = "https://raw.githubusercontent.com/qjbferrer/CPE-019-Sports-Classification-Identifier/main/classes.py"
    response = requests.get(github_url)
    if response.status_code != 200:
        raise Exception("Could not fetch classes.py from GitHub")

    # Execute the code inside classes.py to extract sports_class
    local_vars = {}
    exec(response.text, {}, local_vars)
    class_names = local_vars["sports_class"]

    return model, class_names


model, class_names = load_model_and_classes()

# ---------------------------
# App UI
# ---------------------------
st.title("🏅 Sports Image Classifier")
st.write("Upload a sports image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure image is converted to RGB (fix for shape mismatch)
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    # Preprocess
    img_array = image.img_to_array(img) / 255.0   # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Show results
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Prediction:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Debug (optional)
    with st.expander("🔎 Raw probabilities"):
        st.write(predictions[0])
