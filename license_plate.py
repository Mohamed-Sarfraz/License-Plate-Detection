import numpy as np
from PIL import Image, ImageDraw

def rgb_to_gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

def dilate_horizontal(gray_img):
    dilated = gray_img.copy()
    for i in range(gray_img.shape[0]):
        for j in range(1, gray_img.shape[1] - 1):
            dilated[i, j] = max(gray_img[i, j-1], gray_img[i, j], gray_img[i, j+1])
    return dilated

def compute_histogram(img, axis):
    if axis == 0:
        diff = np.abs(np.diff(img, axis=0))
    else:
        diff = np.abs(np.diff(img, axis=1))
    diff[diff <= 20] = 0
    return np.sum(diff, axis=axis)

def smooth(hist, k=20):
    return np.convolve(hist, np.ones(2*k+1)/(2*k+1), mode='same')

def threshold_filter(hist, avg):
    return (hist >= (avg + np.std(hist) * 0.5)).astype(int)

def find_segments(mask):
    segs = []
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            segs.append((i, j-1))
            i = j
        else:
            i += 1
    return segs

def select_best_region(rows, cols):
    best = (0, 0, 0, 0)
    best_area = 0
    for r0, r1 in rows:
        for c0, c1 in cols:
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 20 or w < 60:
                continue
            ratio = w / h
            area = h * w
            if 2 <= ratio <= 6 and area > best_area:
                best_area = area
                best = (c0, r0, c1, r1)
    return best

def process_image(pil_image):
    img = np.array(pil_image.resize((400, 300)))
    gray = rgb_to_gray(img)
    dilated = dilate_horizontal(gray)

    h_hist = compute_histogram(dilated, axis=0)
    h_hist_smooth = smooth(h_hist)
    h_mask = threshold_filter(h_hist_smooth, np.mean(h_hist_smooth))

    v_hist = compute_histogram(dilated, axis=1)
    v_hist_smooth = smooth(v_hist)
    v_mask = threshold_filter(v_hist_smooth, np.mean(v_hist_smooth))

    h_segs = find_segments(h_mask)
    v_segs = find_segments(v_mask)

    left, top, right, bottom = select_best_region(v_segs, h_segs)

    # Add padding
    padding = 5
    top = max(0, top - padding)
    left = max(0, left - padding)
    right = min(gray.shape[1] - 1, right + padding)
    bottom = min(gray.shape[0] - 1, bottom + padding)

    # Draw rectangle
    boxed_img = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(boxed_img)
    draw.rectangle([left, top, right, bottom], outline="red", width=2)

    return boxed_img, {
        "grayscale": Image.fromarray(gray),
        "dilated": Image.fromarray(dilated)
    }


    return boxed_img, {
        "grayscale": Image.fromarray(gray),
        "dilated": Image.fromarray(dilated)
    }
