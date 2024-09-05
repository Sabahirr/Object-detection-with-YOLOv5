import torch
import streamlit as st
from PIL import Image
import numpy as np

# Set the title of the Streamlit app
st.title("YOLOv5 Object Detection with Custom Model")
st.write("Upload an image and detect objects using a custom-trained YOLOv5 model.")

# Function to load the YOLOv5 model
@st.cache_resource
def load_model():
    # Load the custom-trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_model.pt', force_reload=True)
    return model

# Load the model once and cache it
model = load_model()

# Upload an image file (jpg, jpeg, png)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if the image is uploaded
if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Perform object detection on the image
    results = model(image_np)

    # Display the results on the image
    results.show()  # This will show the detections on the image

    # Render the detections and display the image with the detected objects
    st.image(np.array(results.render()[0]), caption="Detected Objects", use_column_width=True)

