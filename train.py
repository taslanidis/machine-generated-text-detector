import os
import argparse
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from data_engineering.dataset import TextFileDataset

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model_size(model_size: str):
    size_map = {
        # 's': 'FacebookAI/roberta-small',
        'm': 'FacebookAI/roberta-base',
        'l': 'FacebookAI/roberta-large'
    }
    return size_map.get(model_size.lower(), 'FacebookAI/roberta-base')

def train(args):

    # Set random seed
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer + Model
    model_name = get_model_size(args.size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Dataset
    dataset = TextFileDataset(args.data_path, tokenizer, split='train')
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data directory structure and contents.")

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(dataloader) * 50  # 50 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()

            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Average Loss for Epoch {epoch}: {avg_epoch_loss:.4f}")

    # Save the model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = os.path.basename(args.data_path)
    save_path = checkpoint_dir / f"{dataset_name}__{args.size}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

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
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
        default=42
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
        default=50
    )
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()