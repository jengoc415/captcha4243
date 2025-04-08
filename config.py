CONFIG = {
    "model": "cnn",                   # Model type: 'cnn' or 'rnn' (future)
    "batch_size": 64,
    "epochs": 5,
    "learning_rate": 0.001,
    "val_split": 0.2,
    "use_colour": True, # set to true for RGB input
    "use_pretrained": True,
    "image_size": (28, 28),
    "data_path": "dataset/train_letter"
}
