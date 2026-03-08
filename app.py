import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Image Classification App")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("face_recognition_model.h5")
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((224,224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = preprocess_image(image)

    prediction = model.predict(img)

    class_id = np.argmax(prediction)

    st.success(f"Predicted Class: {class_id}")
