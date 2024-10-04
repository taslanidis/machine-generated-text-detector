import openai
from openai import OpenAI
import os
from typing import List, Dict
import re

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-ga3WBWvatxgRyMGvPgjWI11uI2U_K6sWWtybUL_StLT3BlbkFJvSMW78qp6lLmZI0YAmrp0qh9bwjyKISyS_l9T0kbkA"
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGPTInstruct:
    def __init__(self):
        self.client = OpenAI()

    def inference_for_prompt(self, prompts: List[Dict[str, str]]) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that only answer the given question."},
                    {"role": "user", "content": prompt[0]['content']}
                ],
                max_tokens = 200,
                temperature = 0.1
            )
            responses.append(response.choices[0].message.content.strip())
        return responses

class OpenAIGPT:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.client = OpenAI()

    #clean the output text to make it more human like.
    def clean_machine_generated_text(self,text: str) -> str:
        """
        Clean machine-generated text by removing escape characters, normalizing whitespace,
        and handling other machine-specific artifacts.
        """
        # 1. Remove escape characters
        text = text.replace("\n", " ")  # Replace newline escape sequences
        text = text.replace("\'", "'")  # Replace escaped apostrophes
        text = text.replace('\"', '"')  # Replace escaped quotation marks

        # 2. Normalize newlines and whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space

        # 3. Remove excessive newlines or paragraph markers
        text = text.replace('\n', ' ').replace('\r', '')  # Convert any remaining newlines to spaces

        # 4. Fix any leftover escaped characters
        text = text.encode('ascii', 'ignore').decode('unicode_escape')  # Decode and handle any special characters

        # 5. Optionally, handle specific machine patterns (e.g., structured lists)
        text = re.sub(r'\s*\d+\.\s*', '', text)  # Remove patterns like "1. ", "2. ", etc.
        
        # 6. Remove or modify any additional markers that might indicate machine generation
        text = text.strip()  # Trim any leading/trailing spaces

        return text
    def trim_to_last_sentence(self,text: str) -> str:
        """
        Trims the text up to the last complete sentence or phrase ending with a comma, period, or other punctuation.
        This ensures that we don't cut the text off mid-sentence.
        """
        # Use a regular expression to find the last comma or period
        last_punctuation =text.rfind('.') # Get the last comma or period position

        if last_punctuation != -1:  # If a punctuation is found
            return text[:last_punctuation + 1]  # Return the text up to and including the last punctuation
        else:
            return text  # If no punctuation is found, return the original text

    def generate(self, prompts: List[str], temperature: float = 0.8) -> List[str]:
        responses = []
        for prompt in prompts:

            if prompt.startswith("<s>"):
                prompt = prompt.replace("<s>", "START: ")  # Use any other start token you prefer

            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": f"You are a helpful assitant that completes the sentnece to a medium length text around {self.max_length} number of tokens"},
                    {"role": "user", "content": "Complete the following sentence to make a medium length text based on whatever you want related to the start of the sentence. Provide full text included start of the sentence in the prompt. Here is the text:" + prompt}
                ],
                max_tokens = self.max_length,
                temperature = temperature
            )

            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("START:"):
                response_text = response_text.replace("START: ", "")
            response_text = self.clean_machine_generated_text(response_text)
            response_text = self.trim_to_last_sentence(response_text)
            responses.append(response_text)

        return responses 
    
    


