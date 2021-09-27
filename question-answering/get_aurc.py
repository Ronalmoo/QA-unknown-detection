import os
import matplotlib.pyplot as plt
import json
import jsonlines
import numpy as np
import pandas as pd
import string
import re
from sklearn import metrics

from pathlib import Path
from transformers import (
    squad_convert_examples_to_features,
    )
from transformers.data.processors.squad import SquadV1Processor

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""
    
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example - 
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        
    return gold_answers


def extract_q_and_a(factoid_path: Path):
    with factoid_path.open() as json_file:
        data = json.load(json_file)
    
    questions = data['data'][0]['paragraphs']
    data_rows = []

    for question in questions:
        context = question['context']
        for question_and_answer in question['qas']:
            question = question_and_answer['question']
            answers = question_and_answer['answers']

            for answer in answers:
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)

                data_rows.append({
                    'question': question,
                    'context': context,
                    'answer_text': answer_text,
                    'answer_start': answer_start,
                    'answer_end': answer_end,
                })
    return pd.DataFrame(data_rows)


def get_risk_coverage(df: pd.DataFrame) -> float:
    total_questions = len(df)
    total_correct = 0
    covered = 0
    risks = []
    coverages = []

    for em, prob in zip(df['exact'].values, df['prob'].values):
        covered += 1
        if em:
            total_correct += 1
        risks.append(1 - (total_correct / covered))
        coverages.append(covered / total_questions)        
    auc = round(metrics.auc(coverages, risks) * 100, 2)
    
    return auc


def aurc_eaurc(rank_conf, rank_corr):
    li_risk = []
    li_coverage = []
    risk = 0
    for i in range(len(rank_conf)):
        coverage = (i + 1) / len(rank_conf)
        li_coverage.append(coverage)

        if rank_corr[i] == 0:
            risk += 1

        li_risk.append(risk / (i + 1))

    r = li_risk[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in li_risk:
        risk_coverage_curve_area += risk_value * (1 / len(li_risk))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print(f'* AURC\t\t{round(aurc * 100, 2)}')
    print(f'* E-AURC\t{round(eaurc * 100, 2)}')

    return aurc, eaurc


def get_predictions(path: str):
    with open(path) as pred:
        pred_file = json.load(pred)
        preds = []
        probs = []
        # print(pred_file)
        for value in pred_file.values():
            preds.append(value[0]['text'])
            probs.append(value[0]['probability'])
    return preds, probs


def get_goldens(in_goldens, out_goldens):
    in_answers, out_answers = [], []
    for i in range(len(in_goldens)):
        in_answers.append(get_gold_answers(in_goldens[in_qid_to_example_index[in_answer_qids[i]]]))
    for i in range(len(out_goldens)):
        out_answers.append(get_gold_answers(out_goldens[out_qid_to_example_index[out_answer_qids[i]]]))    
    return in_answers, out_answers


def get_exact(in_answers, out_answers):
    in_exact, out_exact = [], []
    for i, _ in enumerate(in_answers):
        if  in_preds[i] in in_answers[i]:
            in_exact.append(1)
        else:
            in_exact.append(0)
        # exact.append(candidated_answers[i] == preds[i])
    # print(exact.count(1) / len(exact))
    for i, _ in enumerate(out_answers):
        if  out_preds[i] in out_answers[i]:
            out_exact.append(1)
        else:
            out_exact.append(0)
        return in_exact, out_exact


def make_df(in_preds, in_probs, out_preds, out_probs, in_exact, out_exact):
    in_df = pd.DataFrame(in_preds)
    in_df['prob'] = pd.DataFrame(in_probs)

    out_df = pd.DataFrame(out_preds)
    out_df['prob'] = pd.DataFrame(out_probs)

    in_df['exact'] = pd.DataFrame(in_exact)
    out_df['exact'] = pd.DataFrame(out_exact)

    return in_df, out_df


if __name__ == "__main__":
    model_list = ['bert-base-cased', 'dmis-lab/biobert-base-cased-v1.1', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'allenai/biomed_roberta_base', 'SpanBERT/spanbert-base-cased']
    ood_list = ["squad", "music", "finance", "film", "law", "computing"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        required=True,
        help="outputs(nbest_predictions) of models: ",
    )
    args = parser.parse_args()
    # path = '/nas/home/moo/NLU/biobert-pytorch/QA-unknown-detection/question-answering/output'

    for domain in ood_list:
        for model in model_list:
            # print(f'{domain} of {model}')
    
            pred_path = args.path + f"/{model}/bioasq_indomain_7b/nbest_predictions_.json"
            squad_pred = args.path + f"/{model}/{domain}_outdomain_7b/nbest_predictions_.json"
            # squad_golden = "/daintlab/data/NLU/squad/dev-v1.1.json"

            processor = SquadV1Processor()
            in_goldens = processor.get_dev_examples("/nas/home/moo/NLU/biobert-pytorch/question-answering/datasets/QA/BioASQ", filename="BioASQ-dev-factoid-7b.json")
            if domain == 'squad':
                out_goldens = processor.get_dev_examples("/daintlab/data/NLU/squad", filename=f"dev-v1.1.json")
            else:
                out_goldens = processor.get_dev_examples("/daintlab/data/NLU/QA/data/marco", filename=f"squad.{domain}.test.json")        

            in_qid_to_example_index = {example.qas_id: i for i, example in enumerate(in_goldens)}
            in_qid_to_has_answer = {example.qas_id: bool(example.answers) for example in in_goldens}

            out_qid_to_example_index = {example.qas_id: i for i, example in enumerate(out_goldens)}
            out_qid_to_has_answer = {example.qas_id: bool(example.answers) for example in out_goldens}

            in_answer_qids = [qas_id for  qas_id, has_answer in in_qid_to_has_answer.items() if has_answer]
            out_answer_qids = [qas_id for qas_id, has_answer in out_qid_to_has_answer.items() if has_answer]

            with open(pred_path) as in_pred:
                in_pred_file = json.load(in_pred)
                in_preds = []
                in_probs = []
                # print(pred_file)
                for value in in_pred_file.values():
                    in_preds.append(value[0]['text'])
                    in_probs.append(value[0]['probability'])
            
            with open(squad_pred) as out_pred:
                out_pred_file = json.load(out_pred)
                out_preds = []
                out_probs = []
                # print(pred_file)
                for value in out_pred_file.values():
                    out_preds.append(value[0]['text'])
                    out_probs.append(value[0]['probability'])
                    
            # check answers
            print(len(in_answer_qids))
            in_answers, out_answers = [], []
            for i in range(len(in_goldens)):
                # print(qid_to_example_index[answer_qids[i]])
                in_answers.append(get_gold_answers(in_goldens[in_qid_to_example_index[in_answer_qids[i]]]))
            
            for i in range(len(out_goldens)):
                # print(qid_to_example_index[answer_qids[i]])
                out_answers.append(get_gold_answers(out_goldens[out_qid_to_example_index[out_answer_qids[i]]]))    
            print(len(out_answers))
            print(len(out_preds))
            # check exact
            in_exact, out_exact = [], []
            for i, _ in enumerate(in_answers):
                if  in_preds[i] in in_answers[i]:
                    in_exact.append(1)
                else:
                    in_exact.append(0)
                # exact.append(candidated_answers[i] == preds[i])
            print(f'ratio of in_exact: {in_exact.count(1) / len(in_exact)}')
            for i, _ in enumerate(out_answers):
                if  out_preds[i] in out_answers[i]:
                    out_exact.append(1)
                else:
                    out_exact.append(0)

            print(f'ratio of out_exact: {out_exact.count(1) / len(out_exact)}')
            in_df = pd.DataFrame(in_preds)
            in_df['prob'] = pd.DataFrame(in_probs)

            out_df = pd.DataFrame(out_preds)
            out_df['prob'] = pd.DataFrame(out_probs)

            in_df['exact'] = pd.DataFrame(in_exact)
            out_df['exact'] = pd.DataFrame(out_exact)
            
            if domain == 'squad':
                concat_df = pd.concat([in_df, out_df.sample(n=len(in_df), random_state=42)], ignore_index=True)
            else:
                concat_df = pd.concat([in_df, out_df], ignore_index=True)
            # concat_df = out_df

            concat_df.sort_values(by='prob', ascending=False, inplace=True)
            concat_df['correct'] = concat_df['prob'][concat_df['exact'] == 1]
            concat_df['unknown'] = concat_df['prob'][concat_df['exact'] == 0]
            
            # Plot
            concat_df.correct.plot.hist(bins=40, alpha=0.3)
            concat_df.unknown.plot.hist(bins=40, alpha=0.3)

            if model == 'bert-base-cased':
                plt.title(f'{model}_{domain}')
            elif model == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
                plt.title(f"PubmedBERT_{domain}")
            else:
                plt.title(f'{model.split("/")[1]}_{domain}')
            
            if not os.path.exists(f'plots/{model}'):
                os.makedirs(f'plots/{model}')

            plt.ylim(([0, 350]))
            plt.legend()
            plt.savefig(f'plots/{model}/{domain}_exact_hist_out.png', dpi=300)
            plt.close()
            
            print(f'{model}_{domain}')
            auc = get_risk_coverage(concat_df)
            # print(concat_df['exact'].loc[concat_df['exact'] == 1])
            print(auc)
            aurc_eaurc(concat_df['prob'].values, concat_df['exact'].values)
