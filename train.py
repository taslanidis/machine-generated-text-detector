import argparse

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from data_engineering.dataset import TextFileDataset

def get_model_size(model_size: str):
    size_map = {
        # 's': 'FacebookAI/roberta-small',
        'm': 'FacebookAI/roberta-base',
        'l': 'FacebookAI/roberta-large'
    }
    return size_map.get(model_size.lower(), 'FacebookAI/roberta-base')

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer + Model
    model_name = get_model_size(args.size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Dataset
    dataset = TextFileDataset(args.data_path, tokenizer)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data directory structure and contents.")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # === DEBUGGING ===

    # first batch
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        print(input_ids)
        print(labels)
        break

    # model architecture
    print(model)

    # === END DEBUGGING ===

def main():
    parser = argparse.ArgumentParser(description="Train a RoBERTa text classifier.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the data folder (e.g., data/my_example_dataset)",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=['s', 'm', 'l'],
        help="Size of model: 's/m/l",
        default='m'
    )
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()