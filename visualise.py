import matplotlib.pyplot as plt
from utils.loader import get_loaders

def visualize_batch():
    loader, _, classes = get_loaders(batch_size=8)
    images, labels = next(iter(loader))

    plt.figure(figsize=(10, 2))
    for i in range(len(images)):
        plt.subplot(1, 8, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_batch()
