import json
import os

def clean_answer_text(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Iterate over each item in the JSON data
    for item in data:
        if 'Answer' in item:
            item['Answer'] = item['Answer'].replace('###Answer###', '').strip()

    # Write the cleaned data back to the file without changing the structure
    with open(file_path, 'w') as file:
        json.dump(data, file)

def main():
    files = ['train.json', 'test.json', 'valid.json']
    for file_name in files:
        file_path = os.path.join('/Users/doruktarhan/Documents/GitHub/machine-generated-text-detector/data/followupqg/openai_qa', file_name)
        if os.path.exists(file_path):
            clean_answer_text(file_path)
            print(f"Cleaned {file_name}")
        else:
            print(f"{file_name} does not exist")

if __name__ == "__main__":
    main()