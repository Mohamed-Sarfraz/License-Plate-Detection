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
    if axis == 0:
        diff = np.abs(np.diff(img, axis=0))
    else:
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

def find_segments(mask):
    segments = []
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            segments.append((i, j-1))
            i = j
        else:
            i += 1
    return segments

def select_best_region(row_segments, col_segments):
    best_score = 0
    best_box = (0, 0, 0, 0)
    for r_start, r_end in row_segments:
        for c_start, c_end in col_segments:
            height = r_end - r_start + 1
            width = c_end - c_start + 1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            if 2 <= aspect_ratio <= 6 and area > best_score:
                best_score = area
                best_box = (c_start, r_start, c_end, r_end)
    return best_box

def find_plate_region(gray_img, row_mask, col_mask):
    row_segments = find_segments(row_mask)
    col_segments = find_segments(col_mask)
    left, top, right, bottom = select_best_region(row_segments, col_segments)
    img_boxed = Image.fromarray(gray_img).convert("RGB")
    draw = ImageDraw.Draw(img_boxed)
    draw.rectangle([left, top, right, bottom], outline="red", width=2)
    return img_boxed

def process_image(pil_image):
    img = np.array(pil_image.resize((400, 300)))
    gray = rgb_to_gray(img)
    dilated = dilate_horizontal(gray)

    col_hist = edge_histogram(dilated, axis=0)
    col_hist_smooth = moving_average(col_hist)
    col_mask = threshold_filter(col_hist_smooth, np.mean(col_hist_smooth))

    row_hist = edge_histogram(dilated, axis=1)
    row_hist_smooth = moving_average(row_hist)
    row_mask = threshold_filter(row_hist_smooth, np.mean(row_hist_smooth))

    boxed_image = find_plate_region(dilated, row_mask, col_mask)

    grayscale = Image.fromarray(gray)
    dilated_img = Image.fromarray(dilated)

    return boxed_image, {
        "grayscale": grayscale,
        "dilated": dilated_img
    }

