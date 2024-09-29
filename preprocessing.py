import argparse

from data_engineering.generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Create LLM generated text.")
    parser.add_argument(
        "--model",
        type=str,
        default="openai",
        help="Model name",
        choices=["llama3", "mistral7b", "gpt2" , "openai" ,"human"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="followupqg",
        help="Dataset name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # pre-processing
    DatasetGenerator.process(
        model=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
