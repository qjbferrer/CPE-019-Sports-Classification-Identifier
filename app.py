import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import predict_label
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('final_model.h5')
  return model
