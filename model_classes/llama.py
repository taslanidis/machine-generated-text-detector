from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
import random
from huggingface_hub import login

# make token on hugging face and insert here
# huggingface_token = "insert_token"
# login(token=huggingface_token)

class LLama3Instruct:

    def __init__(self):
        
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def inference_for_prompt(self, prompts: List[Dict[str, str]]) -> List[str]:
        input_ids = self.tokenizer.apply_chat_template(
            prompts,
            padding=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        prompt_length = input_ids['input_ids'].size(1)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generated_dict = self.model.generate(
            **input_ids,
            eos_token_id=terminators,
            pad_token_id = self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        return decoded_batch


class Llama3:

    def __init__(self, max_length: int):

        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load pre-trained tokenizer and model (e.g., GPT-2)
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
        self.model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def generate(self, human_text: List[str]) -> List[str]:
        """
        Takes a list (batch) of human texts. Uses the first "prime" tokens and generates the rest until max length.

        It returns a list of the relevant machine generated texts.
        """
        # Step 1: Tokenize the batch of human texts
        tokens_batch = self.tokenizer(human_text, return_tensors='pt', padding=True, truncation=True)

        # Step 2: Generate LLM text using the first 10 tokens of each text
        output_batch = self.model.generate(
            tokens_batch['input_ids'].to(self.device), 
            attention_mask=tokens_batch['attention_mask'].to(self.device),
            max_length=self.max_length, 
            do_sample=True, 
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Step 3: Decode the generated output back to text for each batch
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_batch]

        return generated_texts
