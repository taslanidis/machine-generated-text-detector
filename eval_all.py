import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import resource
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

from train import get_model_size

from data_engineering.dataset import JSONDataset
from model import FrozenRoberta


def set_memory_limit(memory_in_gb: float):
    memory = memory_in_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


def evaluate_model_on_dataset(model, dataloader, device):

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > 0).int().view(-1)
            probs = torch.sigmoid(outputs).view(-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, auc


def extract_datasets(data_path, checkpoint_files, tokenizer):
    datasets = {}
    seen = set()
    for checkpoint_file in checkpoint_files:
        folder_name, llm_name = checkpoint_file.stem.split("_")[:2]
        full_path = Path(data_path) / folder_name
        check_path = full_path / llm_name

        if folder_name not in datasets:
            datasets[folder_name] = {}
        if llm_name not in datasets[folder_name]:
            datasets[folder_name][llm_name] = {}

        if check_path not in seen:
            seen.add(check_path)
            train_dataset = JSONDataset(full_path, llm_name, tokenizer, split='train')
            val_dataset = JSONDataset(full_path, llm_name, tokenizer, split='val')
            test_dataset = JSONDataset(full_path, llm_name, tokenizer, split='test')
            datasets[folder_name][llm_name]['train'] = train_dataset
            datasets[folder_name][llm_name]['val'] = val_dataset
            datasets[folder_name][llm_name]['test'] = test_dataset

    return datasets


def in_results(results_df, train_folder_name, train_llm, folder_name, llm_name, size, seed, paired):
    if results_df.empty:
        return False
    w1 = results_df['train_data'] == train_folder_name
    w2 = results_df['train_llm'] == train_llm
    w3 = results_df['test_data'] == folder_name
    w4 = results_df['test_llm'] == llm_name
    w5 = results_df['size'] == size
    w6 = results_df['seed'] == seed
    w7 = results_df['paired'] == paired
    w = w1 & w2 & w3 & w4 & w5 & w6 & w7
    return w.any()


def get_loader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=4,
                      pin_memory=True if torch.cuda.is_available() else False)


def eval_splits(model,
                batch_size,
                train_folder,
                train_llm,
                test_folder,
                test_llm,
                size,
                seed,
                datasets,
                device,
                paired):
    splits = []
    for split in ['train', 'val', 'test']:
        if split != 'test' and not (train_folder == test_folder and train_llm == test_llm):
            continue
        dataset = datasets[test_folder][test_llm][split]
        data_loader = get_loader(dataset, batch_size)
        accuracy, auc = evaluate_model_on_dataset(model, data_loader, device)
        print(f"Model {train_llm}_{size}_{seed}_{paired} evaluated on {test_folder}_{test_llm} ({split}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        splits.append({
            'train_data': train_folder,
            'train_llm': train_llm,
            'test_data': test_folder,
            'test_llm': test_llm,
            'split': split,
            'size': size,
            'seed': seed,
            'paired': paired,
            'accuracy': accuracy,
            'auc': auc
        })
    return splits


def evaluate_all(args):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # List checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    # Load in results, if exists
    results_df = pd.read_csv(args.results_file) if os.path.exists(
        args.results_file) else pd.DataFrame()

    # Define standard model
    model_name = get_model_size(args.size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FrozenRoberta(model_name)
    model.to(device)

    # List all datasets
    datasets = extract_datasets(args.data_path, checkpoint_files, tokenizer)

    # Check if any checkpoints were found
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    if len(datasets) == 0:
        raise ValueError(f"No datasets found")

    results = []
    for checkpoint_file in checkpoint_files:

        # Extract model size and dataset name from checkpoint file name
        train_folder_name, train_llm, size, seed, paired = checkpoint_file.stem.split("_")
        if size != args.size:
            continue
        seed = int(seed.split(".")[0])

        for test_folder, llms in datasets.items():
            for test_llm, dataset in llms.items():
                if in_results(results_df,
                              train_folder_name,
                              train_llm,
                              test_folder,
                              test_llm,
                              size,
                              seed,
                              paired):
                    print(f"Skipping {checkpoint_file.name}")
                    continue

                # Load model
                model.load_parameters(checkpoint_file)
                model.eval()

                new_entries = eval_splits(model,
                                          args.batch_size,
                                          train_folder_name,
                                          train_llm,
                                          test_folder,
                                          test_llm,
                                          size,
                                          seed,
                                          datasets,
                                          device,
                                          paired)

                new_results_df = pd.concat([results_df, pd.DataFrame(new_entries)])
                results_df = new_results_df
                results_df.to_csv(args.results_file, index=False)
                print(f"Results saved (entries: {len(results_df)})")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa text classifiers across multiple datasets.")
    parser.add_argument("--data_path",
                        type=str,
                        default='data/',
                        help="Path to the data folder containing datasets.")
    parser.add_argument("--results_file",
                        type=str,
                        default='results/results.csv',
                        help="Path to the results file.")
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default='checkpoints/',
                        help="Directory containing model checkpoints (.pt files).")
    parser.add_argument("--size",
                        type=str,
                        choices=['s', 'm', 'l'],
                        default='m',
                        help="Size of the RoBERTa model: 's' (small), 'm' (medium), 'l' (large).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()

    set_memory_limit(16)
    evaluate_all(args)


if __name__ == "__main__":
    main()
