import argparse

from data_engineering.instruct_generation import FollowupqgDataset
from data_engineering.generation import GenerativeDataset


def main():
    parser = argparse.ArgumentParser(description="Create LLM generated text.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Model name",
        choices=["llama3", "mistral7b", "human"]
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
    parser.add_argument(
        "--data_type",
        type=str,
        default="json_qa",
        help="Data type"
    )
    args = parser.parse_args()

    # pre-processing
    if args.dataset=='followupqg':

        FollowupqgDataset.process(
            model=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    elif args.dataset in ['squad', 'wikitext']:

        GenerativeDataset.process(
            model=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            max_length=160,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
