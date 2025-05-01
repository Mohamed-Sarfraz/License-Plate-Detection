
import streamlit as st
import numpy as np
from PIL import Image
from license_plate import process_image

st.set_page_config(page_title='License Plate Localization (No Inbuilt Functions)')
st.title('License Plate Localization App')
st.write('**Note:** This app avoids all inbuilt edge detection or image processing functions.')

uploaded_file = st.file_uploader("Upload a car image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Processing...")
    result_img, intermediate_outputs = process_image(image)
    
    st.image(intermediate_outputs['grayscale'], caption='Grayscale Image', use_column_width=True)
    st.image(intermediate_outputs['dilated'], caption='Dilated Image', use_column_width=True)
    st.image(result_img, caption='Final Localized Plate Region', use_column_width=True)
