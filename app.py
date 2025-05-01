
import streamlit as st
import numpy as np
from PIL import Image
from license_plate import process_image

st.set_page_config(page_title='License Plate Localization (No Inbuilt Functions)')
st.title('License Plate Localization (MATLAB Logic in Python)')

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Processing...")
    result, intermediates = process_image(image)

    st.image(intermediates['grayscale'], caption='Grayscale Image')
    st.image(intermediates['dilated'], caption='Dilated Image')
    st.image(result, caption='Final Localized Plate Region')
