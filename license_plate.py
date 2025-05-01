import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Set tesseract path if it's not in the PATH environment variable
# (Windows users need to specify the path to tesseract executable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to detect the number plate from an image
def detect_number_plate(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred_image, 100, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and detect rectangular regions (possible number plates)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Check if the contour is rectangular
            x, y, w, h = cv2.boundingRect(contour)
            # Crop the image to the detected number plate region
            number_plate_region = image[y:y+h, x:x+w]
            
            # Use pytesseract to extract text from the cropped image (number plate)
            number_plate_text = pytesseract.image_to_string(number_plate_region, config='--psm 8')
            return number_plate_region, number_plate_text.strip()
    return None, None

# Streamlit app interface
def main():
    st.title('Car Number Plate Detection')
    
    st.write("Upload a car image to detect the number plate.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Detect number plate
        number_plate_image, number_plate_text = detect_number_plate(image)

        if number_plate_image is not None:
            st.image(number_plate_image, caption="Detected Number Plate", use_column_width=True)
            st.write(f"Detected Number Plate Text: {number_plate_text}")
        else:
            st.write("No number plate detected.")

if __name__ == "__main__":
    main()
