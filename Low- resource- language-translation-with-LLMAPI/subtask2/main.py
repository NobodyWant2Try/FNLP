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

dictionary_path = "./dictionary_za2zh.jsonl"
grammar_path = "./grammar_book.json"
parallel_corpus_path = "./parallel_corpus.json"
test_data_path = "./test_data.json"

grammar_book = utils.get_json_data(grammar_path)
test_data = utils.get_json_data(test_data_path)
parallel_corpus = utils.get_json_data(parallel_corpus_path)
dictionary = utils.get_dictionary(dictionary_path)

results = []
prompts = []

for sample in tqdm(test_data):
    za = sample['za']
    grammar_clues, _ = utils.find_clues_tfidf(grammar_book, za, 60)
    # grammar_clues = grammar_book
    meanings = utils.look_up_dict(za, dictionary)
    parallel_clues = utils.find_parallel(za, parallel_corpus, 140)
    prompt = utils.build_prompt(za, meanings, grammar_clues, parallel_clues)
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
    
with open("submission_task2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "translation"])
    for row in results:
        writer.writerow([row["id"], row["translation"]])

with open("prompt_task2.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)