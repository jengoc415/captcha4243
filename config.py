CONFIG = {
    "model": "cnn_base",                   # Model type: ['cnn_base', 'cnn_pretrained', 'rnn_base']
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 0.001,
    "val_split": 0.2,
    "use_colour": False, # set to true for RGB input
    "image_size": (28, 28),                   # Can be set to None if no resizing is wanted
    "train_path": "dataset/train_letter",
    "test_path": "dataset/test"
}

# CONFIG = {
#     "model": "rnn_base",                  
#     "batch_size": 64,
#     "epochs": 15,
#     "learning_rate": 0.001,
#     "val_split": 0.2,
#     "use_colour": False, 
#     "image_size": (200, 50),  
#     "train_path": "dataset/train",
#     "test_path": "dataset/test"
# }
