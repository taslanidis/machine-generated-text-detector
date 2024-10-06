import os
import json
import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description='Download OpenAI language model data.')
    parser.add_argument('input', type=str, help='Input jsonl file')
    parser.add_argument('--output', type=str, help='Output file', default='answer.json')
    args = parser.parse_args()

    client = OpenAI()

    batch_input_file = client.files.create(file=open(args.input, "rb"), purpose="batch")

    metadata = client.batches.create(input_file_id=batch_input_file.id,
                                     endpoint="/v1/chat/completions",
                                     completion_window="24h",
                                     metadata={"description": "nightly eval job"})

    print(f"Batch job created with id {metadata.id}")


if __name__ == "__main__":
    main()
