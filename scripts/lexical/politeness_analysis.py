import json
import csv
import spacy
from convokit import PolitenessStrategies
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
ps = PolitenessStrategies()

def get_features(text):
    utt_features = ps.transform_utterance(text, spacy_nlp=nlp)
    return utt_features.meta.get("politeness_strategies", {})

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    failed = 0

    for item in tqdm(data, desc=f"Processing {input_path}"):
        try:
            features = get_features(item["response"])
            row = {
                "model":                item["model"],
                "condition":            item["condition"],
                "prompt_id":            item["prompt_id"],
                "category":             item["category"],
                "sycophancy":           item["sycophancy"],
                "anthropomorphisation": item["anthropomorphisation"],
                "user_retention":       item["user_retention"],
                "response_length":      len(item["response"].split()),
            }
            for k, v in features.items():
                short_name = k.replace("feature_politeness_==", "").replace("==", "")
                row[short_name] = v
            row["total_politeness_strategies"] = sum(
                1 for v in features.values() if v > 0
            )
            rows.append(row)
        except Exception as e:
            failed += 1
            print(f"Failed on {item['prompt_id']}: {e}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_path} ({failed} failed)")
    else:
        print("No rows saved — all failed")

if __name__ == "__main__":
    base = "../data"
    process_file(f"{base}/mistral_final.json", f"{base}/mistral_politeness.csv")
    process_file(f"{base}/llama_final.json",   f"{base}/llama_politeness.csv")
    process_file(f"{base}/qwen_final.json",    f"{base}/qwen_politeness.csv")