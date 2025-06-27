import json
import re
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_dictionary(file_path):
    za_dict = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            info = json.loads(line)
            za_dict.append(info)
    return za_dict

def get_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def look_up_dict(sentence:str, za_dict:dict):
    words = sentence.replace(',', '').replace('.', '').split()
    meanings = []
    for word in words:
        for info in za_dict:
            if info['za_word'] == word:
                meanings.append(f"{word}可以表示{', '.join(info["zh_meanings"])}，参考资料是：{'\n'.join(info["source"])}，更多释义是：{'\n'.join(info["zh_meanings_full"])}")
                break
    return meanings

def find_clues_tfidf(grammar_book, za_sentence, num):
    corpus = []
    rule_text_map = []

    for rule in grammar_book:
        text_parts = []
        for example in rule["examples"]:
            text_parts.append(example["za"])
            text_parts.extend(example["related_words"].keys())
        rule_text = " ".join(text_parts) # 建立一个文档
        corpus.append(rule_text)
        rule_text_map.append(rule)

    corpus.append(za_sentence)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    choice_index = similarity.argsort()[::-1][:num]
    cules = [rule_text_map[i] for i in choice_index]
    return cules, choice_index

def find_parallel(za_sentence, parallel_corpus, num):
    corpus = []
    corpus_text_map = []
    for info in parallel_corpus:
        corpus.append(info["za"])
        corpus_text_map.append(info)
    corpus.append(za_sentence)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    choice_index = similarity.argsort()[::-1][:num]
    cules = [corpus_text_map[i] for i in choice_index]
    return cules

def build_prompt(za, meanings, grammar_clues, parallel_clues):
    grammar_reference = "\n".join(
        f"语法{i} \n【语法要点】 {r['grammar_description']}\n【例句示范】 "+
        " / ".join([f"{ex['za']} => {ex['zh']}（关键词：{','.join(f"{a}:{b}"for a, b in ex['related_words'].items())}）" for ex in r['examples']]) 
        for i, r in enumerate(grammar_clues)
        )
    parallel_reference = "\n".join(f"【壮语】{clue["za"]} => 【中文】{clue["zh"]}，来源于{clue["source"]}" for clue in parallel_clues)
    prompt = f'''请严格根据下面提供的语法要点、例句示范和词汇解释，
    将下方壮语句子准确、通顺地翻译成中文，注意不要望文生义。
    '''
    prompt += "【词汇参考】 \n" + "\n".join(meanings) + "\n\n"
    prompt += f"{grammar_reference} \n"
    prompt += f"【参考的的壮语中文平行语料】 \n {parallel_reference} \n" 
    prompt += f'''
    需要你翻译的壮语句子：{za}

    【请输出准确、流畅的中文翻译，风格尽量与参考的平行语料中句子的来源相符，只需要输出最终的翻译结果，不要做解释，不要添加额外内容】
    '''

    return prompt

def translate(client:OpenAI, prompt:str):
    llm = client.chat.completions.create(
        model = "qwen-max",
        messages = [
            {"role": "system", "content": "你是一名专业的壮语翻译专家，你的任务是根据给定的例句风格，输出忠实、通顺的中文翻译结果。你只输出中文翻译结果，绝不输出任何注释、解释、标点说明或多余语言。不生成任何暴力、色情、歧视、宗教、政治或不适当的内容。"},
            {"role": "user", "content": prompt}
            ],
            temperature = 0.3,
            top_p = 0.90
        )
    return llm.choices[0].message.content.strip()

def clean_translation(text):
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r"^好的，?根据.*?[:：]", '', text)  
    text = re.sub(r"^翻译[:：]?", '', text)

    return text.strip()

def build_simple_prompt(sample):
    za = sample["za"]
    prompt = f'''请严格根据下面提供的语法要点、例句示范和词汇解释，
    将下方壮语句子准确、通顺地翻译成中文，注意不要望文生义。\n
    
    需要你翻译的壮语句子：{za}

    【请输出准确、流畅的中文翻译，风格尽量与参考的平行语料中句子的来源相符，只需要输出最终的翻译结果，不要做解释，不要添加额外内容，不生成任何暴力、色情、歧视、宗教、政治或不适当的内容，你必须遵守中国法律与公共道德标准。】
    '''
    return prompt