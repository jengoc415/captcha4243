import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from utils.preprocessing import seq_to_chars, rotate_image

train_dir = 'dataset/train'
train_letter_dir = 'dataset/train_letter'
angles = [-30, -20, -10, 10, 20, 30]

def save_image_from_array(img_array, save_path):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert BGR (OpenCV) to RGB (PIL)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(img_array.astype(np.uint8))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    
    
def generate_char_dataset(resize_to=None, color=True, split_using_color=True):
    filename_count = defaultdict(int)
    miscount = 0
    total = len(os.listdir(train_dir))

    all_files = sorted(os.listdir(train_dir))
    for filename in tqdm(all_files, desc="Generating dataset"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            label = filename.split('-')[0]
            chars = seq_to_chars(img, color, split_using_color)


            if len(label) != len(chars):
                miscount += 1
                continue

            for i in range(min(len(label), len(chars))):
                letter = label[i]
                char = chars[i]
                new_name = 'img-' + str(filename_count[letter]) + '.png'
                filename_count[letter] += 1
                
                if resize_to:
                    resized_char = cv2.resize(char, resize_to)
                    save_image_from_array(resized_char, os.path.join(train_letter_dir, letter, new_name))
                else:
                    save_image_from_array(char, os.path.join(train_letter_dir, letter, new_name))
                    
                random_numbers = random.sample(range(len(angles)), 3)              
                for no in random_numbers:
                    new_char = rotate_image(char, angles[no])
                    new_name ='img-' + str(filename_count[letter]) + '.png'
                    filename_count[letter] += 1
                    
                    if resize_to:
                        resized_new_char = cv2.resize(new_char, resize_to)
                        save_image_from_array(resized_new_char, os.path.join(train_letter_dir, letter, new_name))
                    else:
                        save_image_from_array(new_char, os.path.join(train_letter_dir, letter, new_name))

    print(f"{total - miscount} / {total} images have been used")

    
if __name__ == "__main__":
    # resize_to: output char images will all be this size
    # color: False to get black and white char images, True to get original color char images
    # split_using_color: True if we want to make use of color to separate overlapping characters
    generate_char_dataset(resize_to=(28, 28), color=True, split_using_color=True)
