import os
import matplotlib.pyplot as plt
import cv2
import re

# Insert folder name
train_dir = 'train'
test_dir = 'test'

# Natural sort helper
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_images_from_folder(folder, limit=None):
    images = []
    filenames = []
    count = 0
    for filename in sorted(os.listdir(folder), key=natural_key):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) # load in colour first
            if img is not None:
                images.append(img)
                filenames.append(filename)
                count += 1
                if limit and count >= limit:
                    break
    return images, filenames


# Visualise images
def show_images(images, titles=None):
    plt.figure(figsize=(12,4))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        if titles:
            plt.title(titles[i])
    plt.show()



# Load a few samples to test
train_images, train_filenames = load_images_from_folder(train_dir, limit=5)
test_images, test_filenames = load_images_from_folder(test_dir, limit=5)

# Show images
show_images(train_images, train_filenames)