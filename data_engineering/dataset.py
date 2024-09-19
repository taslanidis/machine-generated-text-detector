import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

LABEL_MAP = {'ai':0, 'human':1}

class TextFileDataset(Dataset):
    """
    A PyTorch Dataset that reads text data from files on-the-fly.
    Assumes a directory structure where each class has its own subdirectory.
    """
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for label_name, label_idx in LABEL_MAP.items():
            label_dir = self.data_dir / label_name
            if not label_dir.is_dir():
                continue
            for file_path in label_dir.glob("*.txt"):
                samples.append((str(file_path), label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # Load the text data from the file
        file_path, label = self.samples[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Squeeze to remove the batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }