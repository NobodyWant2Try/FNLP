import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import jieba

def get_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def find_clues_simple(grammar_book, za_sentence:str, num:int):
    hits = []
    for i, rule in enumerate(grammar_book):
        tokens = set()
        for example in rule["examples"]:
            tokens.update(example["za"].split())
            tokens.update(example["related_words"].keys())
        score = sum(token.lower() in za_sentence for token in tokens)
        hits.append((rule, score, i))
    hits.sort(key=lambda x:x[1], reverse=True)
    clues = [clue for clue, _, i in hits[:num]]
    choice_index = [i for clue, _, i, in hits[:num]]
    return clues, choice_index
    
def find_clues_tfidf(grammar_book, sample, num):
    za_sentence = sample['za']
    za_ralated_words = set(word for value in sample['related_words'].values() for word in jieba.lcut(value))

    corpus = []
    rule_text_map = []
    chinese_similarity = []

    for rule in grammar_book:
        text_parts = []
        chinese_words = set()
        for example in rule["examples"]:
            text_parts.append(example["za"])
            text_parts.extend(example["related_words"].keys())
            chinese_words.update(word for value in example['related_words'].values() for word in jieba.lcut(value))
        rule_text = " ".join(text_parts) # 建立一个文档
        corpus.append(rule_text)
        rule_text_map.append(rule)

    common_words = za_ralated_words & chinese_words
    union_words = za_ralated_words | chinese_words
    jaccard_similarity = len(common_words) / len(union_words) if union_words else 0
    chinese_similarity.append(jaccard_similarity)

    corpus.append(za_sentence)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]


    socres = [(tfidf_score + chinese_score * 0.5, idx) for idx, (tfidf_score, chinese_score) in enumerate(zip(similarity, chinese_similarity))]
    socres.sort(reverse=True)
    choice_index = [idx for _, idx in socres[:num]]
    # choice_index = similarity.argsort()[::-1][:num]
    cules = [rule_text_map[i] for i in choice_index]
    return cules, choice_index

    
def build_prompt(sample, clues):
    za = sample["za"]
    related_words = "\n".join([f"{k} -> {v}" for k, v in sample["related_words"].items()])
    grammar_text = "\n".join(
        f"【语法要点】 {r['grammar_description']}\n【例句示范】 "+
        " / ".join([f"{ex['za']} => {ex['zh']}（关键词：{','.join(f"{a}:{b}"for a, b in ex['related_words'].items())}）" for ex in r['examples']]) 
        for r in clues
        )
    prompt = f"""
    你是一名精通中文和壮语的翻译专家，请严格根据下面提供的语法要点、例句示范和词汇解释，
    将下方壮语句子准确、通顺地翻译成中文，注意不要望文生义。只需要输出最终的翻译结果，不要做解释，不要添加额外内容。
    
    {grammar_text}
    
    【词汇解释】
    {related_words}

    需要你翻译的壮语句子：{za}

    【请输出中文翻译】
    """
    return prompt

def translate(client:OpenAI, prompt:str):
    llm = client.chat.completions.create(
        model = "qwen-max",
        messages = [
            {"role": "system", "content": "你是一名擅长壮语的中文翻译专家。"},
            {"role": "user", "content": prompt}
            ],
            temperature = 0.3,
            top_p = 0.90
        )
    return llm.choices[0].message.content.strip()

def clean_translation(text):
    text = re.sub(r'（.*?）', '', text)
    return text.strip()