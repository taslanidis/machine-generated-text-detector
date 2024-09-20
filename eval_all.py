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
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

from train import seed_everything, get_model_size

from data_engineering.dataset import TextFileDataset

def evaluate_model_on_dataset(model, dataloader, device):

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Apply softmax
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = labels.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    return accuracy, auc

def evaluate_all(args):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # List datasets by checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    # Check if any checkpoints were found
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    datasets = [d for d in Path(args.data_path).glob("*") if d.is_dir()]
    if len(datasets) == 0:
        raise ValueError(f"No datasets found in {args.data_path}")

    results = []


    for checkpoint_file in checkpoint_files:

        # Extract model size and dataset name from checkpoint file name
        train_dataset, model_size = checkpoint_file.stem.split("__")

        # Load the trained model
        model_name = get_model_size(model_size)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.load_state_dict(torch.load(checkpoint_file, map_location=device), strict=True)
        model.to(device)

        # Validate model on each dataset
        for dataset_dir in datasets:
            dataset_name = dataset_dir.name
            val_dataset = TextFileDataset(str(dataset_dir), tokenizer, split="val")
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=False,
                pin_memory=True if torch.cuda.is_available() else False
            )

            # Evaluate
            accuracy, auc = evaluate_model_on_dataset(model, val_dataloader, device)

            print(f"Model {checkpoint_file.name} evaluated on {dataset_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")

            results.append({
                'train': train_dataset,
                'val': dataset_name,
                'size': model_size,
                'model': checkpoint_file.name,
                'accuracy': accuracy,
                'auc': auc
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

    # AUC matrix
    matrix = results_df.pivot(index='model', columns='val', values='auc')
    print("\nCross-evaluation AUC Matrix:")
    print(matrix)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa text classifiers across multiple datasets.")
    parser.add_argument(
        "--data_path",
        type=str,
        default='data/',
        help="Path to the data folder containing datasets."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='checkpoints/',
        help="Directory containing model checkpoints (.pt files)."
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=['s', 'm', 'l'],
        default='m',
        help="Size of the RoBERTa model: 's' (small), 'm' (medium), 'l' (large)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_everything(args.seed)

    evaluate_all(args)

if __name__ == "__main__":
    main()
