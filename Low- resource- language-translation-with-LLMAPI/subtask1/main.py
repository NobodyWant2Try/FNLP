import os
import csv
import json
from tqdm import tqdm
import utils
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "http://dashscope.aliyuncs.com/compatible-mode/v1"   
)

grammar_book = utils.get_json_data("./grammar_book.json")
test_data = utils.get_json_data("./test_data.json")

results = []
prompts = []

for sample in tqdm(test_data):
    simple_clues, index1 = utils.find_clues_simple(grammar_book, sample['za'], 20)
    tfidf_clues, index2 = utils.find_clues_tfidf(grammar_book, sample, 70)
    clues = []
    for clue in simple_clues:
        if clue not in tfidf_clues:
            clues.append(clue)
    clues.extend(tfidf_clues)
    prompt = utils.build_prompt(sample, clues)
    zh = utils.translate(client, prompt)
    zh = utils.clean_translation(zh)
    results.append({
        "id": sample["id"],
        "za": sample["za"],
        "prompt": prompt,
        "translation": zh
        })
    prompts.append({
        "id": sample['id'],
        "prompt": prompt
        })

with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "translation"])
    for row in results:
        writer.writerow([row["id"], row["translation"]])

with open("prompt.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
