# Project 1 Machine Translation for Low-Resource Language

## 文件结构

```file

fnlp_assignment4
│   readme.md
│   report.pdf   
│
└───subtask1
│   │   grammar_book.json
│   │   prompt.json
|   |   prompt_zero_shot.json
│   │   submission.csv
|   |   submission_zero_shot.csv
|   |   main.py
|   |   utils.py
|   |   test_data.json
|   |   zero_shot_experiment.py
│   
│   
└───subtask2
    │   dictionary_za2zh.jsonl
    │   grammar_book.json
    |   parallel_corpus.json
    |   prompt_task2.json
    |   prompt_zero_shot_task2.json
    |   submission_task2.csv
    |   submission_zero_shot_task2.csv
    |   main.py
    |   utils.py
    |   test_data.json
    |   zero_shot_experiment.py
```

subtask1，subtask2文件夹内分别包含每个子任务的python代码，main.py是主函数，utils.py内包含一些辅助函数，zero_shot_experiment.py是与zero-shot base baseline进行比较，验证方法有效性的代码。将从阿里云获取的api配置到环境变量中后，分别运行main.py即可运行两个任务的代码。

## 环境

sutask1中代码在python=3.12环境下运行，需额外安装tqdm，openai，scikit-learn，jieba库。

subtask2中代码在python=3.12环境下运行，需额外安装tqdm，openai，scikit-learn库。

## 运行方式

将Kaggle上下载每个任务的数据集分别移到子任务文件夹下，整理成上面那样的文件结构，在每个子任务文件夹下运行 python main.py即可。要对此方法进行有效性检验，可以运行python zero_shot_experiment.py。
