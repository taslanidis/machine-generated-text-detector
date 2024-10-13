import json
from nltk.translate.bleu_score import sentence_bleu
from comet import download_model, load_from_checkpoint


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


machine_data_paths = [
    "C:/UvA MS AI/DL4NLP/OneDrive_1_2024-10-4/followupqg/llama3_qa/train.json",
    "C:/UvA MS AI/DL4NLP/OneDrive_1_2024-10-4/followupqg/mistral7b_qa/train.json"
]

machine_datasets = [load_dataset(path) for path in machine_data_paths]

comet_model_path = download_model("wmt20-comet-da")
comet_model = load_from_checkpoint(comet_model_path)


def compute_similarity(machine_qa_1, machine_qa_2):
    answer_1 = machine_qa_1["Answer"]
    answer_2 = machine_qa_2["Answer"]

    bleu = sentence_bleu([answer_1.split()], answer_2.split())

    comet_input = [{
        "src": machine_qa_1["Question"],  # Maybe we can also try not contain question info
        "mt": answer_2,
        "ref": answer_1
    }]

    comet_score = comet_model.predict(comet_input, batch_size=1)[0]

    return bleu, comet_score


all_bleu_scores = []
all_comet_scores = []
num_datasets = len(machine_datasets)

for i in range(num_datasets):
    for j in range(i + 1, num_datasets):
        bleu_scores = []
        comet_scores = []

        for idx in range(len(machine_datasets[i])):
            machine_qa_1 = machine_datasets[i][idx]
            machine_qa_2 = machine_datasets[j][idx]

            bleu, comet = compute_similarity(machine_qa_1, machine_qa_2)
            bleu_scores.append(bleu)
            comet_scores.append(comet)

        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        flattened_comet_scores = [score for sublist in comet_scores for score in sublist]
        print(f"comet_scores: ",comet_scores)
        avg_comet = sum(flattened_comet_scores) / len(flattened_comet_scores)
        #avg_comet = sum(comet_scores) / len(comet_scores)

        all_bleu_scores.append((i, j, avg_bleu))
        all_comet_scores.append((i, j, avg_comet))

print("BLEU Similarity Scores between Datasets:")
for i, j, score in all_bleu_scores:
    print(f"Dataset {i + 1} vs Dataset {j + 1}: BLEU Score = {score:.4f}")

print("\nCOMET Similarity Scores between Datasets:")
for i, j, score in all_comet_scores:
    print(f"Dataset {i + 1} vs Dataset {j + 1}: COMET Score = {score:.4f}")
