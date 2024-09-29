import openai
from openai import OpenAI
import os
from typing import List, Dict

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "openai-api-key"
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGPT:
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
                max_tokens = 2000,
                temperature = 0.1
            )
            responses.append(response.choices[0].message.content.strip())
        return responses

