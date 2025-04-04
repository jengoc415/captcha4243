import cv2
import numpy as np

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

def remove_scratch_lines_morphological(binary_img):
    h_kernel = np.ones((1, 3), np.uint8)
    v_kernel = np.ones((3, 1), np.uint8)
    s_kernel = np.ones((2, 2), np.uint8)

    h_opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
    v_opening = cv2.morphologyEx(h_opening, cv2.MORPH_OPEN, v_kernel)
    cleaned = cv2.morphologyEx(v_opening, cv2.MORPH_OPEN, s_kernel)

    closing_kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing_kernel)

def remove_scratch_lines_combined(gray_img, binary_img):
    hough_cleaned = remove_lines_with_hough_enhanced(gray_img, binary_img)
    morph_cleaned = remove_scratch_lines_morphological(hough_cleaned)
    return morph_cleaned

def preprocess_image(img, resize_to=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=3
    )
    cleaned = remove_scratch_lines_combined(gray, binary)
    return cleaned

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    border_value = 255 if len(img.shape) == 2 else (255, 255, 255)
    return cv2.warpAffine(img, rot_matrix, (new_w, new_h), borderValue=border_value)
