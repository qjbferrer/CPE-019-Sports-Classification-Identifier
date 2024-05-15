import streamlit as st
import tensorflow as tf
import numpy as np
import requests

from io import BytesIO
from classNames import class_names
from tensorflow.keras.models import load_model
from PIL import Image
