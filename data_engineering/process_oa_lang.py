import argparse
from pathlib import Path
import os
import json

"""
Notes:

We process both human and ai responses here.

Target output structure:
data/name_of_dataset_groups/oa_lang/human/train.json (done)
data/name_of_dataset_groups/oa_lang/human/val.json (done)
data/name_of_dataset_groups/oa_lang/chatgpt_4o_mini/train.json (needs to be done)
data/name_of_dataset_groups/oa_lang/chatgpt_4o_mini/val.json (needs to be done)

Target file structure:
[{"Question": "What is the capital of France?", "Answer": "Paris"}, ...]

chatgpt jsonl file structure:
{"custom_id": "request-0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a chat bot who answers like a human"}, {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 1000}}

Plan:
1. Read in all jsons from human files.
2. Read in chatgpt batch request jsonl, which includes the ID of the request and match each request question with its file.
3. Read the results.jsonl, and for each ID, match it with the corresponding question and save the answer to the corresponding AI file.
"""

def request_id_to_save_path(root):

    root_path = Path(root)

    # First open jsonl and match each question to an ID
    queston_to_id = {}
    id_to_question = {}
    with open(root_path / 'chatgpt_batch_requests.jsonl', 'r') as f:
        results = [json.loads(x) for x in f]
    for entry in results:
        queston_to_id[entry['body']['messages'][1]['content']] = entry['custom_id']
        id_to_question[entry['custom_id']] = entry['body']['messages'][1]['content']

    # List all subfolders
    languages = [x for x in os.listdir(root_path) if os.path.isdir(root_path / x)]

    # Read in all human responses
    human_paths = {}
    for lang in languages:
        for split in ['train', 'val']:
            file_path = root_path / lang / 'human' / f'{split}.json'
            with open(file_path, 'r') as f:
                qa_pairs = json.load(f)
            questions = [x['Question'] for x in qa_pairs]
            for question in questions:
                human_paths[queston_to_id[question]] = file_path

    return human_paths, id_to_question

def save_to_json(results, id_to_path, id_to_question):

    files_to_save = {}
    for entry in results:

        i = entry['custom_id']

        if i not in id_to_path:
            continue

        
        if id_to_path[i] not in files_to_save:
            files_to_save[id_to_path[i]] = []

        files_to_save[id_to_path[entry['custom_id']]].append(
            {
                "Question": id_to_question[i],
                "Answer": entry['response']['body']['choices'][0]['message']['content'],
            }
        )

    for file, data in files_to_save.items():
        file = str(file).replace('human', 'chatgpt_4o_mini')
        os.makedirs(Path(file).parent, exist_ok=True)
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Process OpenAI language model data.')
    parser.add_argument('ai_response', type=str, help='Input ai response jsonl file')
    parser.add_argument('root', type=str, help='The root directory of the dataset')
    args = parser.parse_args()

    # Read ID to path mapping
    id_to_path, id_to_question = request_id_to_save_path(args.root)

    # Read in all ai responses
    with open(args.ai_response, 'r') as f:
        results = [json.loads(x) for x in f]

    # Save responses to the correct file
    save_to_json(results, id_to_path, id_to_question)

if __name__ == "__main__":
    main()