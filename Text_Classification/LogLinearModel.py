from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import utils
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

class LogLinearModel():
    def __init__(self, dataname):
        self.dataname = dataname
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.pipe = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=150000)),
                # ('len', utils.AdditionalFeature()) # 在训练Hallmarks of Cancer Corpus数据集时可供选择使用，更多的解释见Report
            ])),
            ('clf', LogisticRegression(penalty='l2', C = np.sqrt(5), max_iter=1000))
        ])

    def get_data(self):
        if self.dataname == '20newsgroups':
            self.dataset = load_dataset('SetFit/20_newsgroups')
            self.train_data = self.dataset['train']
            self.test_data = self.dataset['test']
            self.X_train = self.train_data['text']
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
    
    def train(self):
        self.pipe.fit(self.X_train, self.Y_train)
        return

    def test(self, results):
        Y_pred = self.pipe.predict(self.X_test)
        Y_pred_train = self.pipe.predict(self.X_train)
        results['loglinear'][self.dataname] = {
            "train": utils.estimation(self.Y_train, Y_pred_train),
            "test": utils.estimation(self.Y_test, Y_pred)
        }
        return results

