
import numpy as np
from PIL import Image

def rgb_to_gray(img_array):
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def dilate_horizontal(gray_img):
    rows, cols = gray_img.shape
    dilated = gray_img.copy()
    for i in range(rows):
        for j in range(1, cols-1):
            dilated[i, j] = max(gray_img[i, j-1], gray_img[i, j], gray_img[i, j+1])
    return dilated

def edge_histogram(img, axis=0):
    if axis == 0:  # horizontal
        diff = np.abs(np.diff(img, axis=0))
    else:  # vertical
        diff = np.abs(np.diff(img, axis=1))
    diff[diff <= 20] = 0
    return np.sum(diff, axis=axis)

def moving_average(hist, k=20):
    smoothed = np.copy(hist)
    for i in range(k, len(hist) - k):
        smoothed[i] = np.mean(hist[i-k:i+k+1])
    return smoothed

def threshold_filter(hist, avg):
    mask = hist >= avg
    return mask.astype(int)

def mask_image(gray_img, row_mask, col_mask):
    masked = gray_img.copy()
    for i in range(gray_img.shape[0]):
        if row_mask[i] == 0:
            masked[i, :] = 0
    for j in range(gray_img.shape[1]):
        if col_mask[j] == 0:
            masked[:, j] = 0
    return masked

def process_image(pil_image):
    img = np.array(pil_image.resize((400, 300)))  # Resize for simplicity
    gray = rgb_to_gray(img)
    dilated = dilate_horizontal(gray)
    
    # Horizontal edge processing
    col_hist = edge_histogram(dilated, axis=0)
    col_hist_smooth = moving_average(col_hist)
    col_mask = threshold_filter(col_hist_smooth, np.mean(col_hist_smooth))
    
    # Vertical edge processing
    row_hist = edge_histogram(dilated, axis=1)
    row_hist_smooth = moving_average(row_hist)
    row_mask = threshold_filter(row_hist_smooth, np.mean(row_hist_smooth))
    
    # Final Mask
    final = mask_image(dilated, row_mask, col_mask)
    
    result = Image.fromarray(final)
    grayscale = Image.fromarray(gray)
    dilated_img = Image.fromarray(dilated)
    
    return result, {
        "grayscale": grayscale,
        "dilated": dilated_img
    }
