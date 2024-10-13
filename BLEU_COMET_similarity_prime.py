import json
from nltk.translate.bleu_score import sentence_bleu
from comet import download_model, load_from_checkpoint


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


dataset_1_path = "C:/UvA MS AI/DL4NLP/OneDrive_1_2024-10-4/squad/llama3_text/val.json"
dataset_2_path = "C:/UvA MS AI/DL4NLP/OneDrive_1_2024-10-4/wikitext/openai_text/val.json"

dataset_1 = load_dataset(dataset_1_path)
dataset_2 = load_dataset(dataset_2_path)

comet_model_path = download_model("wmt20-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

def compute_similarity(text_1, text_2):
    bleu = sentence_bleu([text_1.split()], text_2.split())

    comet_input = [{
        "src": "",
        "mt": text_2,
        "ref": text_1
    }]
    comet_score = comet_model.predict(comet_input, batch_size=1)[0]

    return bleu, comet_score

bleu_scores = []
comet_scores = []

for idx in range(min(1000, len(dataset_1), len(dataset_2))):

    text_1 = dataset_1[idx]["text"]
    text_2 = dataset_2[idx]["text"]

    bleu, comet = compute_similarity(text_1, text_2)
    bleu_scores.append(bleu)
    comet_scores.append(comet)

avg_bleu = sum(bleu_scores) / len(bleu_scores)
flattened_comet_scores = [score for sublist in comet_scores for score in sublist]
print(f"comet_scores: ",comet_scores)
avg_comet = sum(flattened_comet_scores) / len(flattened_comet_scores)

# Output results
print(f"Average BLEU Score between datasets: {avg_bleu:.4f}")
print(f"Average COMET Score between datasets: {avg_comet:.4f}")
