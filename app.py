import streamlit as st
import tensorflow as tf
import numpy as np
import requests

from io import BytesIO
from classes import sports_class
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("best_model.h5")
