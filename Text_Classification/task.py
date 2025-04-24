import json
import torch
from LogLinearModel import LogLinearModel
from BERTModel import BERTModel

if __name__ == "__main__":
    print("Running LogLinear Model...")
    results = {"loglinear": {}, "bert": {}}
    
    print("Training 20-Newsgroups...")
    model1 = LogLinearModel("20newsgroups")
    model1.get_data()
    model1.train()
    results = model1.test(results)
    
    print("Training HoC...")
    model1 = LogLinearModel("HoC")
    model1.get_data()
    model1.train()
    results = model1.test(results)
    
    print("Running BERT Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training 20-Newsgroups...")
    model2 = BERTModel("20newsgroups", device, 3, 20)
    model2.get_data()
    model2.Train()
    results = model2.get_result(results)

    print("Training HoC...")
    model2 = BERTModel("HoC", device, 5, 11)
    model2.get_data()
    model2.Train()
    results = model2.get_result(results)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
