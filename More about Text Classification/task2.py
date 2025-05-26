from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# step 1 Train WordPiece Tokenizer
file_path = "pubmed_sampled_corpus.jsonline"
vocab_size = 30000

def load_corpus():
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            yield obj['text'] # 逐个处理，防止爆内存

trained_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
trained_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
trained_tokenizer.train_from_iterator(load_corpus(), trainer=trainer)

# step 2 Update Bert Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab = list(trained_tokenizer.get_vocab().keys())

# print("The size of resulting vocabulary: ", len(vocab))

new_tokens = [token for token in vocab if token not in bert_tokenizer.get_vocab()]
# print(len(new_tokens))
new_tokens = sorted(new_tokens, key=lambda x: (-len(x), x))
domain_specific_tokens = [token for token in new_tokens if token not in ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]]
domain_specific_tokens = domain_specific_tokens[:5000]

# select_tokens = random.sample(domain_specific_tokens, 50)
# print(select_tokens)

bert_tokenizer.add_tokens(domain_specific_tokens)
bert_tokenizer.save_pretrained("./expanded_BERT_tokenizer")
# step 3 Update BertModel
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=11).to(device)
model.resize_token_embeddings(len(bert_tokenizer))

# step 4 Train on HoC
def preprocess(data, tokenizer):
    return tokenizer(data['text'], truncation=True)

def estimation(y_label, y_pred):
    accuracy = accuracy_score(y_label, y_pred)
    macro_f1 = f1_score(y_label, y_pred, average='macro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')
    return {
        "accuracy": "{:.4f}".format(round(accuracy, 4)),
        "macro_f1": "{:.4f}".format(round(macro_f1, 4)),
        "micro_f1": "{:.4f}".format(round(micro_f1, 4))
    }

def compute_metrics(eval_prediction):
    prediction, y_label = eval_prediction
    y_pred = prediction.argmax(axis=-1)
    return estimation(y_label, y_pred)

train_data = pd.read_parquet("./data/HoC/train.parquet")
test_data = pd.read_parquet("./data/HoC/test.parquet")
X_train = train_data['text'].tolist()
Y_train = np.ravel(train_data['label'].tolist())
X_test = test_data['text'].tolist()
Y_test = np.ravel(test_data['label'].tolist())
train_dataset = Dataset.from_dict({"text": X_train, "label": Y_train}).map(lambda x: preprocess(x, bert_tokenizer), batched=True)
test_dataset = Dataset.from_dict({"text": X_test, "label": Y_test}).map(lambda x: preprocess(x, bert_tokenizer), batched=True)

args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.05,
    output_dir=f"./BertModel/",
    fp16=True,
    optim="adamw_torch"
    
)
data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer) # 确保批次中所有样本具有相同长度

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics, data_collator=data_collator)
trainer.train()
# 记录结果
results = {"train": {}, "test": {}}

train_score = trainer.evaluate(train_dataset)
test_score = trainer.evaluate(test_dataset)
results['train'] = {
    "accuracy": train_score['eval_accuracy'],
    "macro_f1": train_score['eval_macro_f1'],
    "micro_f1": train_score['eval_micro_f1']
}
results['test'] = {
    "accuracy": test_score['eval_accuracy'],
    "macro_f1": test_score['eval_macro_f1'],
    "micro_f1": test_score['eval_micro_f1']
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)