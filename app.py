import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import predict_label
from PIL import Image

# Title and description
st.title("Sports Image Classification")
st.write("Predict the sport that is being represented in the image.")

# Load Keras model with caching
@st.cache(allow_output_mutation=True)
def load_keras_model():
    model = load_model('final_model.h5')
    return model

# Load the model
model = load_keras_model()

# Function for form and prediction
def import_and_predict(image, model):
    label = predict_label(image, model)
    return label

# Form for image upload and prediction
with st.form("my_form"):
    uploaded_file = st.file_uploader("Upload an image of a sport being played:", type="jpg")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            label = import_and_predict(image, model)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"<h2 style='text-align: center;'>{label}</h2>", unsafe_allow_html=True)
        else:
            st.write("Please upload an image or choose a sample image.")

# Button for using a sample image
st.write("If you would not like to upload an image, you can use the sample image instead:")
sample_img_choice = st.button("Use Sample Image")

if sample_img_choice:
    ima
