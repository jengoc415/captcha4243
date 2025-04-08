# CAPTCHA Character Recognition

This project builds a character-level CAPTCHA recognition system using a CNN trained on custom segmented and preprocessed CAPTCHA images.

---

## Project Structure

```text
captcha_project/
├── dataset/
│   ├── train/                  # Original full CAPTCHA images
│   └── train_letter/           # Output folder with cropped characters (by label)
│       ├── A/
│       ├── B/
│       └── ...
│
├── models/
│   └── cnn.py                  # CNN model definition (CNN, RNN,etc.)
│   └── pretrained_cnn.py
│
├── utils/
│   ├── preprocessing.py        # Full preprocessing pipeline (line removal, cropping, etc.)
│   ├── data_generation.py      # Generates dataset by processing full CAPTCHAs
│   ├── transforms.py           # Torchvision transforms (normalize, resize, augment)
│   └── loader.py               # Custom Dataset class + PyTorch DataLoader
│
├── visualise.py                # Visualise preprocessed characters
├── train.py                    # Training loop for CNN model
├── config.py                   # Centralised config for paths, hyperparams, etc.
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview + instructions
```

## 🧪 Requirements

- Python 3.7+
- `torch`, `torchvision`
- `opencv-python`
- `matplotlib`
- `Pillow`
- `numpy`

## Install with:

pip install -r requirements.txt

## Step 1: Preprocess CAPTCHA Images

Run the following script to generate labeled character images from our original CAPTCHA dataset.

python data_generation/generate_dataset.py

This will:

- Preprocess each CAPTCHA image (grayscale, denoise, remove scratch lines)
- Segment into individual characters
- Save them into dataset/train_letter/{label}/img-XXX.png
- Augment each character with slight rotations

## Step 2: Train the CNN

Start training with:

python train.py

Training uses:

- CNN defined in models/cnn.py
- Dataset loaded via utils/loader.py
- Configurable settings in config.py

Example training output:

```text
Epoch [1/10], Loss: 2.34, Accuracy: 36.5%
...
Epoch [10/10], Loss: 1.24, Accuracy: 74.3%
```

## Step 3: Configuration (via config.py)

You can tune:

```text
CONFIG = {
"model": "cnn",
"batch_size": 64,
"epochs": 25,
"learning_rate": 0.001,
"image_size": (28, 28),
"data_path": "dataset/train_letter"
}
```
