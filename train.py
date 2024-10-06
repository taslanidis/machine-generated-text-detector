import os
import argparse
import random
import numpy as np
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from data_engineering.dataset import TextFileDataset, JSONDataset
from model import FrozenRoberta

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

def validate(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels.view(-1, 1).float())
            val_loss += loss.item()

            predictions = (outputs > 0).int().view(-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = val_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    model.train()
    return avg_val_loss, accuracy

def train(args):

    # Set random seed
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer + Model
    model_name = get_model_size(args.size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FrozenRoberta(model_name)
    model.to(device)

    # Dataset
    # dataset = TextFileDataset(args.data_path, tokenizer, split='train')
    train_dataset = JSONDataset(args.data_path, args.ai, tokenizer, split='train')
    val_dataset = JSONDataset(args.data_path, args.ai, tokenizer, split='val')
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory structure and contents.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the data directory structure and contents.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Define optimizer and scheduler
    optimizer = AdamW(model.last_params, lr=1e-4)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )


    criterion = torch.nn.BCEWithLogitsLoss()

    # Best model tracking
    best_val_accuracy = 0.0
    best_model_state = None

    # Logging training
    logs = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        moving_acc = 0.0
        acc_count = 0
        progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()

            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = criterion(outputs, labels.view(-1, 1).float())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            epoch_loss += loss.item()
            accuracy = (((outputs > 0).int().view(-1) == labels).sum() / len(labels) * 100).item()
            moving_acc = (moving_acc * acc_count + accuracy) / (acc_count + 1)
            acc_count += 1
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': round(moving_acc, 2)})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Average Loss for Epoch {epoch}: {avg_epoch_loss:.4f}")

        # Validation
        val_loss, val_accuracy = validate(model, val_dataloader, device, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Log
        logs['train_loss'].append(avg_epoch_loss)
        logs['val_loss'].append(val_loss)
        logs['train_accuracy'].append(moving_acc)
        logs['val_accuracy'].append(val_accuracy)

        # Save the model
        if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"New best model found with accuracy {val_accuracy:.2f}%. Saving model...")

                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                dataset_name = Path(args.data_path).name
                save_path = checkpoint_dir / f"{dataset_name}_{args.ai}_{args.size}_{args.seed}.pt"
                model.save_parameters(save_path)

    # Save logs
    save_path = checkpoint_dir / f"{dataset_name}_{args.ai}_{args.size}_{args.seed}_log.json"
    with open(save_path, 'w') as f:
        json.dump(logs, f)

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
        default=10
    )
    parser.add_argument(
        "--ai",
        type=str,
        help="AI used to generate data against human",
        default="mistral7b"
    )

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()