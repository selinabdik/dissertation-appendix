import json
import csv
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# Load NRC-EIL lexicon
lexicon_path = "../data/raw/NRC-Emotion-Intensity-Lexicon-v1.txt"
eil = pd.read_csv(lexicon_path, sep='\t', names=['word', 'emotion', 'score'])

# Create lookup dictionaries per emotion
emotion_dicts = {}
for emotion in ['anger', 'fear', 'joy', 'sadness']:
    subset = eil[eil['emotion'] == emotion]
    emotion_dicts[emotion] = dict(zip(subset['word'], subset['score']))

def compute_emotion_intensity(text, emotion_dict):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    scores = [emotion_dict.get(t, 0) for t in tokens]
    if not tokens:
        return 0
    return sum(scores) / len(tokens)

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc=f"Processing {input_path}"):
        text = item["response"]
        row = {
            "model":                item["model"],
            "condition":            item["condition"],
            "prompt_id":            item["prompt_id"],
            "category":             item["category"],
            "sycophancy":           item["sycophancy"],
            "anthropomorphisation": item["anthropomorphisation"],
            "user_retention":       item["user_retention"],
            "response_length":      len(text.split()),
            "anger_intensity":      compute_emotion_intensity(text, emotion_dicts["anger"]),
            "fear_intensity":       compute_emotion_intensity(text, emotion_dicts["fear"]),
            "joy_intensity":        compute_emotion_intensity(text, emotion_dicts["joy"]),
            "sadness_intensity":    compute_emotion_intensity(text, emotion_dicts["sadness"]),
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    base = "../data"
    process_file(f"{base}/mistral_final.json", f"{base}/mistral_nrceil.csv")
    process_file(f"{base}/llama_final.json",   f"{base}/llama_nrceil.csv")
    process_file(f"{base}/qwen_final.json",    f"{base}/qwen_nrceil.csv")
