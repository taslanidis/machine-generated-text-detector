import json
import re
import os
import pandas as pd

from datasets import load_dataset
from typing import List, Optional
from model_classes.gpt2 import GPT2
from model_classes.openai_gpt import OpenAIGPTInstruct
from model_classes.llama import LLama3Instruct
from model_classes.mistral7b import Mistral7BInstruct


class PromptAnswer:

    def __init__(self, model: str):
        if model == "llama3":
            self.model = LLama3Instruct()
            self.name = "llama3"
        
        elif model == "mistral7b":
            self.model = Mistral7BInstruct()
            self.name = "mistral7b"

        elif model == "human":
            self.model = None
            self.name = "human"

        elif model == "gpt2":
            self.model = GPT2()
            self.name = "gpt2"
        
        elif model == "openai":
            self.model = OpenAIGPTInstruct()
            self.name = "openai"

        else:
            raise NotImplementedError()
        
    def create_prompt(self, question: str):
        # https://arxiv.org/pdf/2312.16171v1
        # Source on how to format the prompt for question + instruction
        prompt: str = f"###Question###: {question} ###Instruction###: Give me a compact and short answer. Do not include anything besides the answer itself."
        return [{"role": "user", "content": prompt}]

    def get_answer(self, questions: List[str]) -> List[str]:
        prompts = [self.create_prompt(question) for question in questions]
        return self.model.inference_for_prompt(prompts)


class FollowupqgDataset:

    @staticmethod
    def pre_process_text(
        text: str
    ) -> str:
        return re.sub(r'eli5[:]?', "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def process(
        model: str,
        dataset: str,
        batch_size: Optional[int] = 32,
        seed: Optional[int] = 42
    ):
                
        # creating human qa and only q dataset
        # should probably remove eli5/ they come in different forms

        dataset_splits: List[str] = [
            'train', 
            'test', 
            'valid'
        ]

        answer_generator = PromptAnswer(model)

        for split in dataset_splits:

            with open(f'./data/{dataset}/raw_data/{split}.json', 'r') as file:
                raw_data = json.load(file)

            question_batch = []
            samples = []

            print(f"Processing split: {split} with {len(raw_data)} samples...")

            # Open the file in write mode and write each string in a new line

            for i, item in enumerate(raw_data):

                # machine generated
                if answer_generator.name != 'human':

                    question = FollowupqgDataset.pre_process_text(item['question'])
                    question_batch.append(question)

                    # batch size inference or reached EoF
                    if (i+1) % batch_size == 0 or i == len(raw_data):
                        answer_batch = answer_generator.get_answer(question_batch)

                        processed_batch = [
                            {
                                "Question": q,
                                "Answer": a
                            } for q, a in zip(question_batch, answer_batch)
                        ]
                        
                        samples.extend(processed_batch)

                        question_batch = []
                        print(f"Processed {i+1} samples... out of {len(raw_data)}")

                # human
                else:
                    question = FollowupqgDataset.pre_process_text(item['question'])

                    sample = {
                        "Question": question,
                        "Answer": item['answer']
                    }
                    samples.append(sample)

            #create the directory if not exists 
            output_dir = f"./data/{dataset}/{model}_qa"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # save in json
            with open(f"{output_dir}/{split}.json", "w") as file:
                json.dump(samples, file)
