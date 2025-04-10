import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from models.cnn import SimpleCNN, PretrainedCNN
from models.rnn import CNNLSTMCTC
from utils.dataset import get_img_dataset, get_test_dataset, collate_fn
from train import CNN_MODELS, RNN_MODELS
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

CONFIG = None

CLASSES = 36
PRINT_EVERY = 50  # For clean SLURM logs

def is_running_under_slurm():
    return 'SLURM_JOB_ID' in os.environ

def load_model(device):
    model_filename = CONFIG["model"]
    if CONFIG["use_colour"]:
        model_filename += "_color"
    model_path = os.path.join("trained_models", model_filename + ".pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found!\n"
            f"Please run 'train.py' first to train and save the model."
        )

    if CONFIG["model"] in CNN_MODELS:
        in_channels = 3 if CONFIG["use_colour"] else 1
        if CONFIG["model"] == 'cnn_base':
            model = SimpleCNN(num_classes=CLASSES, in_channels=in_channels)
        elif CONFIG["model"] == 'cnn_pretrained':
            model = PretrainedCNN(num_classes=CLASSES)
        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

    elif CONFIG["model"] in RNN_MODELS:
        in_channels = 3 if CONFIG["use_colour"] else 1
        if CONFIG["model"] == 'rnn_base':
            model = CNNLSTMCTC(CLASSES, in_channels=in_channels)
        else:
            raise ValueError(f"Unknown model '{CONFIG['model']}'")

    else:
        raise ValueError(f"Unknown model '{CONFIG['model']}'")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully from '{model_path}'")
    return model

def greedy_decoder(logits, idx_to_char):
    pred = torch.argmax(logits, dim=2)  # (T, B)
    pred = pred.permute(1, 0)  # (B, T)

    results = []
    for p in pred:
        seq = []
        prev = -1
        for char_idx in p:
            if char_idx.item() != prev and char_idx.item() != 0:
                seq.append(idx_to_char[char_idx.item()])
            prev = char_idx.item()
        results.append("".join(seq))
    return results

def evaluate(model, device):
    total_words = 0
    correct_words = 0
    total_chars = 0
    correct_chars = 0
    y_true = []
    y_pred = []

    use_tqdm = not is_running_under_slurm()

    if CONFIG["model"] in CNN_MODELS:
        print("Loading test dataset...")
        test_dataset = get_test_dataset(CONFIG['test_path'], CONFIG['use_colour'], CONFIG['image_size'], CONFIG['train_path'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    elif CONFIG["model"] in RNN_MODELS:
        print("Loading test dataset...")
        test_dataset = get_img_dataset(CONFIG['test_path'], CONFIG['use_colour'], CONFIG['image_size'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    else:
        raise ValueError(f"Unknown model '{CONFIG['model']}'")

    print("Evaluating model...")
    with torch.no_grad():
        loop = test_loader
        if use_tqdm:
            loop = tqdm(loop, desc="Evaluating", ncols=100, file=sys.stdout)

        for batch_idx, batch in enumerate(loop):
            if CONFIG["model"] in CNN_MODELS:
                char_images, label = batch
                label = label[0]
                predictions = []
                for char_image in char_images:
                    char_image = char_image.to(device)
                    pred = model(char_image)
                    pred_idx = pred.argmax(dim=1).item()
                    pred_char = test_dataset.idx_to_char[pred_idx]
                    predictions.append(pred_char)

                predicted_str = "".join(predictions)

            elif CONFIG["model"] in RNN_MODELS:
                images, labels, _ = batch
                images = images.to(device)
                logits = model(images)
                predictions = greedy_decoder(logits.cpu(), test_dataset.idx_to_char)
                assert len(predictions) == 1, "Batch size must be 1 for this evaluation code!"
                predicted_str = predictions[0]
                label = "".join([test_dataset.idx_to_char[idx.item()] for idx in labels])

            total_words += 1
            if predicted_str == label:
                correct_words += 1

            char_matches = 0
            for true_c, pred_c in zip(label, predicted_str):
                y_true.append(test_dataset.char_to_idx[true_c])
                y_pred.append(test_dataset.char_to_idx.get(pred_c, -1))

                if true_c == pred_c:
                    char_matches += 1

            correct_chars += char_matches
            total_chars += len(label)

            if not use_tqdm and batch_idx % PRINT_EVERY == 0:
                word_acc = correct_words / total_words * 100
                char_acc = correct_chars / total_chars * 100
                print(f"Batch {batch_idx} | Word Acc: {word_acc:.2f}% | Char Acc: {char_acc:.2f}%")
                sys.stdout.flush()

            if use_tqdm:
                word_acc = correct_words / total_words * 100
                char_acc = correct_chars / total_chars * 100
                loop.set_postfix(word_acc=f"{word_acc:.2f}%", char_acc=f"{char_acc:.2f}%")

    # Final Metrics
    word_acc = correct_words / total_words * 100
    char_acc = correct_chars / total_chars * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

    print(f"Word Accuracy: {word_acc:.2f}%")
    print(f"Character Accuracy: {char_acc:.2f}%")
    print(f"Character-level Macro Precision: {precision:.2f}%")
    print(f"Character-level Macro Recall: {recall:.2f}%")
    print(f"Character-level Macro F1 Score: {f1:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        CONFIG = json.load(f)

    print("========== EVALUATE.PY START ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    evaluate(model, device)
