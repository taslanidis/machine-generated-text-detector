from pathlib import Path
from typing import List, Tuple
import json
import torch
import random

from torch.utils.data import Dataset
from transformers import AutoTokenizer


LABEL_MAP = {'human':0, 'ai':1}

class TextFileDataset(Dataset):
    
    def __init__(self, data_dir: str, tokenizer : AutoTokenizer, max_length: int = 512, split : str = 'train') -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for label_name, label_idx in LABEL_MAP.items():
            label_dir = self.data_dir / self.split / label_name
            if not label_dir.is_dir():
                continue
            for file_path in label_dir.glob("*.txt"):
                samples.append((str(file_path), label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:

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

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class JSONDataset(Dataset):

    def __init__(self, data_dir: str, ai: str, tokenizer: AutoTokenizer, max_length: int = 256, split: str = 'train', max_size: int = 12800) -> None:
        self.data_dir = Path(data_dir)
        self.ai = ai
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.max_size = max_size
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        classes = ['human', self.ai]

        # Look up directories and map them. E.g. human -> human_qa
        directories = {d.name.split("_")[0]: d.name for d in self.data_dir.iterdir() if d.is_dir()}
        count = {class_name: 0 for class_name in classes}

        for class_name in classes:
            file_path = self.data_dir / directories[class_name] / f"{self.split}.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            is_qa = "qa" in directories[class_name]
            first_key = "Question" if is_qa else "prime"
            second_key = "Answer" if is_qa else "text"
            for entry in data:
                text = f"###Input: {entry[first_key]}\n\n ###Output: {entry[second_key]}"
                label = 0 if class_name == 'human' else 1
                if count[class_name] >= self.max_size // 2:
                    continue
                count[class_name] += 1
                samples.append((text, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

class UnpairedJSONDataset(JSONDataset):

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        classes = ['human', self.ai]

        # Look up directories and map them. E.g. human -> human_qa
        directories = {d.name.split("_")[0]: d.name for d in self.data_dir.iterdir() if d.is_dir()}
        count = {class_name: 0 for class_name in classes}
        initialized_indexes: bool = False
        human_datapoints: list = []
        
        for class_name in classes:
            file_path = self.data_dir / directories[class_name] / f"{self.split}.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not initialized_indexes:
                n: int = len(data)
                datapoint_list = list(range(n+1))
                random.shuffle(datapoint_list)
                human_datapoints = datapoint_list[:n//2]
                initialized_indexes = True
            
            is_qa = "qa" in directories[class_name]
            first_key = "Question" if is_qa else "prime"
            second_key = "Answer" if is_qa else "text"
            for i, entry in enumerate(data):
                
                # some primes (datapoints), must be seen only by human, some other only by ai
                if class_name == "human" and i not in human_datapoints:
                    continue

                elif class_name == self.ai and i in human_datapoints:
                    continue
                
                text = f"###Input: {entry[first_key]}\n\n ###Output: {entry[second_key]}"
                label = 0 if class_name == 'human' else 1
               
                if count[class_name] >= self.max_size // 2:
                    continue
                
                count[class_name] += 1
                samples.append((text, label))
        return samples