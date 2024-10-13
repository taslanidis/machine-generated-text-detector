from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# make token on hugging face and insert here
huggingface_token = "INSERT_YOUR_TOKEN"
login(token=huggingface_token)

class GPT2:
    model_id: str = 'gpt2'

    def __init__(self):
        """ if torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu' """
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        

    def inference_for_prompt(self, prompts) -> List[str]:
        # Encode the prompts and pad them
        
        plain_text_prompts = [item[0]['content'] for item in prompts] 

        encodeds = self.tokenizer(
            plain_text_prompts, 
            return_tensors='pt', 
            padding=True
        )
        prompt_length = encodeds['input_ids'].size(1)
        model_inputs = {key: value.to(self.device) for key, value in encodeds.items()}
        
        generated_dict = self.model.generate(
            **model_inputs,
            max_new_tokens=encodeds['input_ids'].size(1),
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True, 
            return_dict_in_generate=True
        )
        generated_ids = generated_dict.sequences
        # Decode tokens starting from the index after prompt length for each prompt in the batch
        decoded_batch = [self.tokenizer.decode(generated_ids[i][prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        return decoded_batch