from transformers import BertTokenizer
from datasets import Dataset
import random
import pandas as pd
import numpy as np

# 加载HoC数据集，从中采样3句话，分别用原始bert和expanded bert进行分词
# 比较训练集分词后的平均长度

train_data = pd.read_parquet("./data/HoC/train.parquet")
X_train = train_data['text'].tolist()
Y_train = np.ravel(train_data['label'].tolist())
train_dataset = Dataset.from_dict({"text": X_train, "label": Y_train})

original_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
expanded_tokenizer = BertTokenizer.from_pretrained("./expanded_BERT_tokenizer")

# 采样句子分词：
sample = random.sample(X_train, 3)
for i, sentence in enumerate(sample):
    print(sentence, '\n')
    original_results = original_tokenizer.tokenize(sentence)
    expanded_results = expanded_tokenizer.tokenize(sentence)
    print("original: ", original_results, '\n')
    print("expanded: ", expanded_results, '\n')
    print("--------------------------")

print("--------------------------")
# 比较训练集分词后的平均token长度
def func(tokenizer, data):
    sum = 0
    for x in data:
        tokens = tokenizer.tokenize(x['text'])
        sum += len(tokens)
    return sum/len(data)

print("original average length: ", func(original_tokenizer, train_dataset))
print("expanded average length: ", func(expanded_tokenizer, train_dataset)) 
