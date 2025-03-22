import os
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np

# Folder setup
train_dir = 'train'
test_dir = 'test'

# Natural sort helper
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Image loader
def load_images_from_folder(folder, limit=None):
    images = []
    filenames = []
    count = 0
    for filename in sorted(os.listdir(folder), key=natural_key):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(filename)
                count += 1
                if limit and count >= limit:
                    break
    return images, filenames

# Hough line removal
def remove_lines_with_hough_enhanced(gray_img, binary_img):
    equalized = cv2.equalizeHist(gray_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    enhanced = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(enhanced, 30, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=10)
    cleaned = binary_img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            if 10 < abs(angle) < 170:
                cv2.line(cleaned, (x1, y1), (x2, y2), 255, 2)
    return cleaned

#  Morphological scratch line removal 
def remove_scratch_lines_morphological(binary_img):
    h_kernel = np.ones((1, 3), np.uint8)
    v_kernel = np.ones((3, 1), np.uint8)
    s_kernel = np.ones((2, 2), np.uint8)
    
    h_opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
    v_opening = cv2.morphologyEx(h_opening, cv2.MORPH_OPEN, v_kernel)
    cleaned = cv2.morphologyEx(v_opening, cv2.MORPH_OPEN, s_kernel)

    closing_kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing_kernel)
    
    return cleaned

# Combined approach
def remove_scratch_lines_combined(gray_img, binary_img):
    hough_cleaned = remove_lines_with_hough_enhanced(gray_img, binary_img)
    morph_cleaned = remove_scratch_lines_morphological(hough_cleaned)
    return morph_cleaned

# Preprocessing pipeline
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=3
    )

    cleaned = remove_scratch_lines_combined(gray, binary)
    return gray, binary, cleaned

# Visualize all 5 images
def show_all_results(images, filenames):
    plt.figure(figsize=(15, 18))
    for i, img in enumerate(images):
        gray, binary, cleaned = preprocess_image(img)

        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{filenames[i]} - Original")
        plt.axis('off')

        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary")
        plt.axis('off')

        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(cleaned, cmap='gray')
        plt.title("Final Output")
        plt.axis('off')

    plt.suptitle("Scratch Line Removal Results for CAPTCHA (0–4)", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# Run
train_images, train_filenames = load_images_from_folder(train_dir, limit=5)
show_all_results(train_images, train_filenames)
