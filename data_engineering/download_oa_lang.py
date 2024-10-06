import os
import sys
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import resource
from datasets import load_dataset, DatasetDict, concatenate_datasets

def is_debugging():
    if sys.gettrace() is not None:
        return True
    if os.getenv('PYDEVD_USE_FRAME_EVAL') is not None:
        return True
    return False

def set_memory_limit(memory_in_gb: float):
    memory = memory_in_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

def extract_conversation(dataset):
    """Extracting the first two messages in each conversation."""

    # Combine train + val
    dataset = concatenate_datasets([dataset['train'], dataset['validation']])

    if is_debugging():
        dataset = dataset.select(range(5000))
        print("Debugging mode: Only using 2000 samples.")

    conversations = []

    # Find entries with no parent id
    def accepted(entry):
        w1 = entry['parent_id'] is None
        w2 = entry['review_result']
        w3 = ~entry['deleted']
        w4 = ~entry['synthetic']
        w = w1 and w2 and w3 and w4
        return w
    first_messages = {entry['message_id']: entry for entry in dataset if accepted(entry)}

    # Now let's build the conversations
    for entry in tqdm(dataset):
        if entry['parent_id'] is not None and entry['parent_id'] in first_messages:
            first_message = first_messages[entry['parent_id']]['text']
            second_message = entry['text']
            new_entry = {
                'request': first_message,
                'response': second_message,
                'lang': entry['lang']
            }
            conversations.append(new_entry)

            # Remove the first message from the available list
            del first_messages[entry['parent_id']]
    
    return conversations

def filter_languages(conversations, topk=5):
    """
    Keep only the top-k languages with the most conversations.
    """
    language_counts = defaultdict(int)
    
    # Count entries
    for conv in conversations:
        language_counts[conv['lang']] += 1
    
    # Sort
    top_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:topk]
    print(f"Top languages: {top_languages}")

    return [lang for lang, _ in top_languages]

def sample_conversations_by_language(conversations, languages):
    """Sample 70% for training and 30% for validation."""
    language_samples = {lang: {'train': [], 'val': []} for lang in languages}
    
    for lang in languages:
        lang_conversations = [conv for conv in conversations if conv['lang'] == lang]

        n = len(lang_conversations)
        train_n = int(n * 0.7)
        val_n = n - train_n

        # Random split
        # random.shuffle(lang_conversations)
        language_samples[lang]['train'] = lang_conversations[:train_n]
        language_samples[lang]['val'] = lang_conversations[train_n:train_n+val_n]
    
    return language_samples

def save_conversations_to_json(language_samples, output_dir):
    
    for lang, samples in language_samples.items():

        lang_dir = Path(output_dir) / lang / "human"
        os.makedirs(lang_dir, exist_ok=True)

        save_to_json(samples['train'], lang_dir / "train.json")
        save_to_json(samples['val'], lang_dir / "val.json")

def save_to_chatgpt_batch_req_jsonl(conversations, save_path):

    formatted_data = []
    seen = set()
    for idx, conversation in enumerate(conversations):

        n_words = len(conversation['request'].split())
        identifier = f"{conversation['request']} {conversation['response']}"

        entry = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": f"You are a chat bot who answers like a human. " + \
                                                  f"You answer in about {n_words} words."},
                    {"role": "user", "content": conversation['request']},
                ],
                "max_tokens": 1000,
            }
        }
        formatted_data.append(entry)
    
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # Batches start with a .jsonl file where each line contains the details of an individual request to the API.
    with open(save_path, "w") as f:
        for entry in formatted_data:
            json.dump(entry, f)
            f.write("\n")
    
    print(f"Saved {len(conversations)} entries to {save_path}")


def save_to_json(conversations, save_path):

    formatted_data = []
    for idx, conversation in enumerate(conversations):
        entry = {
            "Question": conversation['request'],
            "Answer": conversation['response']
        }
        formatted_data.append(entry)

    with open(save_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Saved {len(conversations)} entries to {save_path}")

def main(args):

    set_memory_limit(16)

    print("Downloading dataset...")
    dataset = load_dataset("OpenAssistant/oasst1")
    
    print("Extracting conversations...")
    all_conversations = extract_conversation(dataset)

    print("Filtering languages...")
    languages = filter_languages(all_conversations)
    
    print("Sampling train and validation data...")
    sampled_conversations = sample_conversations_by_language(all_conversations, languages)

    print("Saving data to JSON files...")
    save_conversations_to_json(sampled_conversations, args.output_dir)
    
    print("Saving ChatGPT batch request JSONL files...")
    save_path = Path(args.output_dir) / "chatgpt_batch_requests.jsonl"
    save_to_chatgpt_batch_req_jsonl(all_conversations, save_path)

    
    print("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OpenAssistant Dataset for Multilingual Training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/open_assistant_lang/",
        help="Directory to save the processed data."
    )
    args = parser.parse_args()

    main(args)
