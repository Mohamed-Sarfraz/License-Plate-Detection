import numpy as np
from PIL import Image, ImageDraw

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

def find_plate_region(gray_img, row_mask, col_mask):
    row_indices = np.where(row_mask == 1)[0]
    col_indices = np.where(col_mask == 1)[0]
    if len(row_indices) == 0 or len(col_indices) == 0:
        return Image.fromarray(gray_img)
    top, bottom = row_indices[0], row_indices[-1]
    left, right = col_indices[0], col_indices[-1]
    img_boxed = Image.fromarray(gray_img).convert("RGB")
    draw = ImageDraw.Draw(img_boxed)
    draw.rectangle([left, top, right, bottom], outline="red", width=2)
    return img_boxed

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
    
    # Extract probable plate region
    boxed_image = find_plate_region(dilated, row_mask, col_mask)
    
    grayscale = Image.fromarray(gray)
    dilated_img = Image.fromarray(dilated)
    
    return boxed_image, {
        "grayscale": grayscale,
        "dilated": dilated_img
    }
