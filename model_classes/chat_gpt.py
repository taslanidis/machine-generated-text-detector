from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import torch
from huggingface_hub import login



class Chat_gpt:

    def __init__(self):
        
        self.model_id ='path'

    def inference(self, input):

        output= 0

        return output