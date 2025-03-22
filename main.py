import os
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np

# Insert folder name
train_dir = 'train'
test_dir = 'test'

# Natural sort helper to help sort images alphabetically
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

# Binarise 1 image
def binarise_image(img):
    # Convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding - calculates diff thresholds for local regions. Otsu was ineffective.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return binary

# Visualise original + binarised image side by side
def show_binarisation_result(original, binary, title="Binarisation"):
    plt.figure(figsize=(16,8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap="gray")
    plt.title("Binarised")
    plt.axis("off")

    plt.suptitle(title)
    plt.show()


# Load a few samples to test. Remove the limits if you want. Limit auto set to 0 if you dont include it as param.
train_images, train_filenames = load_images_from_folder(train_dir, limit=5)
test_images, test_filenames = load_images_from_folder(test_dir, limit=5)

# Show images
img_choice = 0
binary_sample = binarise_image(train_images[img_choice])
sample = train_images[img_choice]
sample_title = train_filenames[img_choice]
show_binarisation_result(sample, binary_sample, sample_title)