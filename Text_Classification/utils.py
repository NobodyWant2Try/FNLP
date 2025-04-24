from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AdditionalFeature(BaseEstimator, TransformerMixin):
    # 统计每条文本的词数和长度
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_len = len(text)
            words_len = len(text.split())
            features.append([text_len, words_len])
        features = np.array(features)
        return features

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
    predictions, y_label = eval_prediction
    y_pred = predictions.argmax(axis=-1)
    return estimation(y_label, y_pred)

def preprocess(data, tokenizer):
    # 预处理raw data
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=512)
