import os
import cv2
import random
from collections import defaultdict
from PIL import Image
from utils.preprocessing import preprocess_image, rotate_image

angles = [-30, -20, -10, 10, 20, 30]
filename_count = defaultdict(int)

def save_image_from_array(img_array, save_path):
    img = Image.fromarray(img_array.astype('uint8'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def get_letters(cleaned_img, min_area=50):
    flipped = cv2.bitwise_not(cleaned_img)
    contours, _ = cv2.findContours(flipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes = sorted(boxes, key=lambda b: b[0])

    letters = [cleaned_img[y:y+h, x:x+w] for (x, y, w, h) in boxes]
    return letters

def generate_char_dataset(train_dir='dataset/train', output_dir='dataset/train_letter'):
    count, miscount = 0, 0
    for filename in sorted(os.listdir(train_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        label = filename.split('-')[0]
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        cleaned = preprocess_image(img)
        chars = get_letters(cleaned)

        if len(label) != len(chars):
            miscount += 1

        for i in range(min(len(label), len(chars))):
            letter = label[i]
            char = cv2.resize(chars[i], (28, 28))
            new_name = f'img-{filename_count[letter]}.png'
            filename_count[letter] += 1
            save_image_from_array(char, os.path.join(output_dir, letter, new_name))

            for angle in random.sample(angles, 3):
                rotated = rotate_image(char, angle)
                new_name = f'img-{filename_count[letter]}.png'
                filename_count[letter] += 1
                save_image_from_array(rotated, os.path.join(output_dir, letter, new_name))

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images")

    print(f"Done. Mismatched character counts: {miscount}")
