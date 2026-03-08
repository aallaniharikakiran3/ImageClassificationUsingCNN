import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🔍 Anomaly Detection using ResNet + Autoencoder")

st.write("Upload an image to check whether it is Normal or Anomalous.")

# Load model
@st.cache_resource
def load_model():
    model = torch.load("face_recognition_model.h5", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    # reconstruction error
    loss = torch.mean((img - output) ** 2).item()

    threshold = 0.02   # adjust based on training

    st.write("### Reconstruction Error:", loss)

    if loss > threshold:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("✅ Normal Image")