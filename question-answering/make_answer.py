import json
import numpy as np
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":

    with open('/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-test-factoid-7b_answers.json') as f:
        file = json.load(f)
    # print(file['data'][0]['paragraphs'][0]['context'])
    
    for data in file['data']:
        # print(data['paragraphs'])
        del data['paragraphs']
        for context in data['paragraphs']:
]
            for qa in context['qas']:
                for text in qa['answers']:
                    qa['answers'] = [{"text": text}]

                # print(qa)
    
    with open("0427_test.json", "w") as json_file:
                    json.dump(file, json_file, indent=2)

