import pickle
import joblib
import os
import torch

# Load your pre-trained model
model = joblib.load("mobilenet.model")
model.eval()



import streamlit as st
from PIL import Image
import numpy as np

with open(r"class_labels.txt", 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
def preprocess(image):
    # Custom preprocessing logic
    return image.resize((640, 640))
with open(r"custom_model (1)-Copy1.pt", 'rb') as file:
    Ymodel = torch.load(file)
Ymodel.eval()
    

import torchvision.transforms as transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])



# Streamlit UI

my_page = st.sidebar.radio('MENU', ['Image Class Prediction', 'Image bounding'])
if my_page == 'Image Class Prediction':
 st.title("Image Prediction App")
 st.write("Upload an image, and the model will predict its class!")

# File uploader
 uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

 if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Processing...")
    processed_image = preprocess(image)
    final_image=processed_image .unsqueeze(0)
    prediction = model(final_image)
    preds = (prediction > 0.5).float()
    if preds.item()==1:
        st.write(f"Predicted Class is DRONE")
    else:
        st.write(f"Predicted Class is BIRD")

    st.write(f"Confidence Score is {prediction.item()} ")

elif my_page=="Image bounding":
  st.title("Image bounding")
  st.write("Image bounding")

  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Processing...")
    
    results = Ymodel.predict(source=image, save=False)
    annotated_image = results[0].plot()
    annotated_image = Image.fromarray(annotated_image)
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)




    
