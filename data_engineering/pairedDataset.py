from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import json


class PairedQADataset(Dataset):

    def __init__(self, human_data_path: str, machine_data_path: str, tokenizer: AutoTokenizer, paired: bool = False,
                 max_length: int = 512) -> None:
        self.human_data_path = Path(human_data_path)
        self.machine_data_path = Path(machine_data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()
        self.paired = paired

    def _load_samples(self) -> List[Tuple[str, str, int]]:
        samples = []

        # Load human and then machine data
        with open(self.human_data_path, 'r', encoding='utf-8') as f:
            human_data = json.load(f)

        with open(self.machine_data_path, 'r', encoding='utf-8') as f:
            machine_data = json.load(f)

        # To use this dataloader we need to ensure "Datasets must have the same number of entries in this case."
        assert len(human_data) == len(machine_data), "Datasets must have the same number of entries in this case."

        if self.paired:
            # For each question, pair each human answer with the corresponding machine-generated answer
            for idx in range(len(human_data)):
                question = human_data[idx]['Question']
                human_answer = human_data[idx]['Answer']
                machine_answer = machine_data[idx]['Answer']

                # Add corresponding labels
                samples.append((question, human_answer, 1))
                samples.append((question, machine_answer, 0))

        else:
            # For each human Q&A, we don't pair it with the corresponding machine-generated answer
            for idx in range(len(human_data)):
                question_1 = human_data[idx]['Question']
                human_answer = human_data[idx]['Answer']

                question_2 = human_data[-idx-1]['Question']
                machine_answer = machine_data[-idx-1]['Answer']

                # Add corresponding labels
                samples.append((question_1, human_answer, 1))
                samples.append((question_2, machine_answer, 0))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        question, answer, label = self.samples[idx]

        text = f"Question: {question} [SEP] Answer: {answer}"

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