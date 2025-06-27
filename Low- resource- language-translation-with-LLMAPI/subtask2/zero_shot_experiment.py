import os
import csv
import json
import utils
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "http://dashscope.aliyuncs.com/compatible-mode/v1"
)

test_data = utils.get_json_data("./test_data.json")

results = []
prompts = []

for sample in tqdm(test_data):
    za = sample['za']
    prompt = utils.build_simple_prompt(sample)
    zh = utils.translate(client, prompt)
    zh = utils.clean_translation(zh)
    results.append({
        "id": sample["id"],
        "translation": zh
        })
    prompts.append({
        "id": sample["id"],
        "prompt": prompt
        })
    
with open("submission__zero_shot_task2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "translation"])
    for row in results:
        writer.writerow([row["id"], row["translation"]])

with open("prompt_zero_shot_task2.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)