import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from license_plate import process_image  # Importing the image processing function from license_plate.py

# Streamlit app title
st.title("Car Number Plate Detection")

# File uploader to upload an image
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Process the image to detect the number plate
    number_plate_image, number_plate_text = process_image(image)

    # Display results
    if number_plate_image is not None:
        st.image(number_plate_image, caption="Detected Number Plate", use_column_width=True)
        st.write(f"Detected Number Plate Text: {number_plate_text}")
    else:
        st.write("No number plate detected.")
