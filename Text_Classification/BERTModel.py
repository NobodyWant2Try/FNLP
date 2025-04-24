import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #放在其他import之前，防止tokenizers库的并行机制与pytorch的多线程冲突警报
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import utils
import numpy as np

class BERTModel():
    def __init__(self, dataname, device, num_epoch, num_classification=None):
        self.dataname = dataname
        self.data = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.num_classification = num_classification
        self.device = device
        self.num_epoch = num_epoch
        self.trainer = None
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_classification).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.args = TrainingArguments(
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=self.num_epoch,
            weight_decay=0.01,
            output_dir=f"./BERTModel/{self.dataname}",
            fp16=True
        )
    
    def get_data(self):
        if self.dataname == '20newsgroups':
            self.data = load_dataset('SetFit/20_newsgroups')
            self.train_data = self.data['train']
            self.test_data = self.data['test']
            self.X_train  = self.train_data['text']
            self.Y_train = self.train_data['label']
            self.X_test = self.test_data['text']
            self.Y_test = self.test_data['label']
        elif self.dataname == 'HoC': 
            self.train_data = pd.read_parquet("./data/HoC/train.parquet")
            self.test_data = pd.read_parquet("./data/HoC/test.parquet")
            self.X_train = self.train_data['text'].tolist()
            self.Y_train = np.ravel(self.train_data['label'].tolist())
            self.X_test = self.test_data['text'].tolist()
            self.Y_test = np.ravel(self.test_data['label'].tolist())
        else:
            pass
        return
        
    def Train(self):
        # map用于把预处理的数据解码，tokenizer化
        # Dataset不支持.to(device)在train时会自动自动到GPU上
        self.train_data = Dataset.from_dict({"text": self.X_train, "label": self.Y_train}).map(lambda x: utils.preprocess(x, self.tokenizer), batched=True)
        self.test_data = Dataset.from_dict({"text": self.X_test, "label": self.Y_test}).map(lambda x: utils.preprocess(x, self.tokenizer), batched=True)
        self.trainer = Trainer(model=self.model, args=self.args, train_dataset=self.train_data, eval_dataset=self.test_data, compute_metrics=utils.compute_metrics)
        self.trainer.train()
        return
        
    def get_result(self, results):
        train_score = self.trainer.evaluate(self.train_data)
        test_score = self.trainer.evaluate(self.test_data)
        # print(train_score)
        # train_score{'eval_loss','eval_accuracy','eval_macro_f1','eval_micro_f1','eval_runtime','eval_samples_per_second','eval_steps_per_second'}
        results["bert"][self.dataname] = {
            "train": {
                "accuracy": train_score['eval_accuracy'],
                "macro_f1": train_score['eval_macro_f1'],
                "micro_f1": train_score['eval_micro_f1']
            },
            "test": {
                "accuracy": test_score['eval_accuracy'],
                "macro_f1": test_score['eval_macro_f1'],
                "micro_f1": test_score['eval_micro_f1']
            }
        }
        return results