import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import predict_label
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('final_model.h5')
  return model

def import_and_predict(image_data, model):
        size = (150,150)  
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image, dtype = 'float32')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img / 255
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        #img_reshape = img[np.newaxis,...]
        prediction = model.predict(img)
        return prediction
