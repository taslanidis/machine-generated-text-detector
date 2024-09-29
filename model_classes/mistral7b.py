from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login

# make token on hugging face and insert here
# huggingface_token = "insert_token"
# login(token=huggingface_token)

class Mistral7BInstruct:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def inference_for_prompt(self, prompts) -> List[str]:
        # Encode the prompts using the chat template and pad them
        encodeds = self.tokenizer.apply_chat_template(prompts, return_tensors='pt', padding=True, return_dict=True)
        prompt_length = encodeds['input_ids'].size(1)
        model_inputs = encodeds.to(self.device)
        generated_dict = self.model.generate(
            **model_inputs,
            max_new_tokens=300,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        return decoded_batch
    



class Mistral7B:

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load pre-trained tokenizer and model (e.g., GPT-2)
        self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
        self.model = AutoModelForCausalLM.from_pretrained(
            'mistralai/Mistral-7B-v0.3',
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
