import os
import json
import re
import pandas as pd
import time

from datasets import load_dataset
from typing import List, Optional
from model_classes.llama import Llama3
from model_classes.mistral7b import Mistral7B
from model_classes.openai_gpt import OpenAIGPT

LOW_THRESH: int = 20
HIGH_THRESH: int = 1000


class TextGenerator:

    def __init__(self, model: str, max_length: int):
        if model == "llama3":
            self.model = Llama3(max_length)
            self.name = "llama3"
        
        elif model == "mistral7b":
            self.model = Mistral7B(max_length)
            self.name = "mistral7b"
        
        elif model == "human":
            self.model = None
            self.name = "human"
        
        elif model == "openai":
            self.model = OpenAIGPT(max_length)
            self.name = "openai"
        else:
            raise NotImplementedError()
        
    def get_text(self, human_text: List[str]) -> List[str]:
        return self.model.generate(human_text)



class GenerativeDataset:

    @staticmethod
    def process(
        model: str,
        dataset: str,
        batch_size: Optional[int] = 32,
        max_length: Optional[int] = 60,
        seed: Optional[int] = 42
    ):

        text_generator = TextGenerator(model, max_length=max_length)

        for split in ['train', 'val', 'test']:
            data_df = pd.read_csv(f'./data/{dataset}/{split}.csv')

            samples = []

            # Step 1: Process the DataFrame in batches
            time1 = time.time()
            for start_idx in range(0, len(data_df), batch_size):
                batch_df = data_df['prompt'].iloc[start_idx:start_idx + batch_size]
                answer_df = data_df['answer'].iloc[start_idx:start_idx + batch_size]

                if text_generator.name != "human":
                    generated_texts = text_generator.get_text(batch_df.to_list())
                else:
                    generated_texts = [prime + " " + answer for prime, answer in zip(batch_df.to_list(), answer_df.to_list())]

                for prime, generated_text in zip(batch_df.to_list(), generated_texts):
                    samples.append(
                        {
                            "prime": prime,
                            "text": generated_text
                        }
                    )
                time2 = time.time()
                print('Processed number of batches:', start_idx, 'out of', len(data_df))
                print(f'Total time passed: {round(time2 - time1,2)} seconds')
                if start_idx > 0:
                    break

            output_dir = f"./data/{dataset}/{model}_text"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # save in json
            with open(f"{output_dir}/{split}.json", "w") as file:
                json.dump(samples, file)
