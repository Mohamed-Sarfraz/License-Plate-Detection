import cv2
import pytesseract
import numpy as np

# Set tesseract path if it's not in the PATH environment variable (Windows example)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(image):
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

