import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import argparse

def check_answer(pred_path, golden_path):
     with open(pred_path) as pred:
        pred_file = json.load(pred)
        preds = []
        for i in pred_file.keys():
            preds.append(pred_file[i][0]['text'])
       

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

    print(f'* AURC\t\t{round(aurc * 1000, 2)}')
    print(f'* E-AURC\t{round(eaurc * 1000, 2)}')

    return aurc, eaurc


def tpr95(ind_confidences, ood_confidences):
    #calculate the falsepositive error when tpr is 95%
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    fprBase = fpr / total

    return fprBase


def get_curve(in_scores, out_scores, stypes=['Baseline']):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = in_scores
        novel = out_scores

        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        fpr_at_tpr95[stype] = fp[stype][tpr95_pos] / num_n
    tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
    fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
    return tp, fp, tpr, fpr, fpr_at_tpr95
    # return tp, fp, fpr_at_tpr95


def load_ind_probs(path: str):
    with open(path) as in_domain:
        in_file = json.load(in_domain)
        in_probabilities = []
        for i in in_file.keys():
            in_probabilities.append(in_file[i][0]['probability'])

        in_probs = np.array(in_probabilities)
        ind_labels = np.ones(len(in_probs))
    
    return in_probs, ind_labels


def load_ood_probs(path: str):
    with open(path) as out_domain:
        out_file = json.load(out_domain)
        out_probabilities = []
        for i in out_file.keys():
            out_probabilities.append(out_file[i][0]['probability'])

        out_probs = np.array(out_probabilities)
        ood_labels = np.zeros(len(out_probs))
    
    return out_probs, ood_labels


def make_label(ind_labels: np.array, ood_labels: np.array):
    labels = np.concatenate((ind_labels, ood_labels))
    return labels
    

def make_probs(ind_probs: np.array, ood_probs: np.array):
    probs = np.concatenate((ind_probs, ood_probs))
    return probs


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        required=True,
        help="outputs(nbest_predictions) of models: ",
    )
    args = parser.parse_args()

    stype = ['baseline']
    ood_list = ["squad", "music", "finance", "biomedical", "film", "law", "computing"]
    model_list = ['bert-base-cased', 'dmis-lab/biobert-base-cased-v1.1', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'allenai/biomed_roberta_base', 'SpanBERT/spanbert-base-cased']
    
    # path = '/nas/home/moo/NLU/biobert-pytorch/QA-unknown-detection/question-answering/output'

    for ood_data in ood_list:
        for model in model_list:
            in_path = args.path + f'/{model}/bioasq_indomain_7b/nbest_predictions_.json'
            out_path = args.path + f'/{model}/{ood_data}_outdomain_7b/nbest_predictions_.json'

            in_probs, ind_labels = load_ind_probs(in_path)
            out_probs, ood_labels = load_ood_probs(out_path)
            # in_probs = scaler.fit_transform(in_probs.reshape(-1, 1))
            print(f"length of in_labels: {len(ind_labels)}")
            print(f"length of out_labels:{len(ood_labels)}")

            labels = make_label(ind_labels, ood_labels)
            probs = make_probs(in_probs, out_probs)            
            in_probs_df = pd.DataFrame(in_probs)
            out_probs_df = pd.DataFrame(out_probs)

            if not os.path.exists(f'description/{model}'):
                os.makedirs(f'description/{model}')
            in_probs_df.describe().to_csv(f'description/{model}/in_probs.csv', index=True)
            out_probs_df.describe().to_csv(f'description/{model}/{ood_data}_out_probs.csv', index=True)

            # histogram
            if ood_data == 'squad':
                prob_concat_df = pd.concat([in_probs_df, out_probs_df.sample(len(in_probs), random_state=42)], axis=1, ignore_index=False)
            else:
                prob_concat_df = pd.concat([in_probs_df, out_probs_df], axis=1, ignore_index=False)
            prob_concat_df.columns = ["in_prob", "out_prob"]

            # print(prob_concat_df)

            # histogram
            prob_concat_df.in_prob.plot.hist(bins=40, alpha=0.3, linewidth=0.00001)
            prob_concat_df.out_prob.plot.hist(bins=40, alpha=0.3, linewidth=0.00001)

            if model == 'bert-base-cased':
                plt.title(f"Model: {model}, OOD data: " + ood_data)
            elif model == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
                plt.title(f"Model: PubmedBERT, OOD data: " + ood_data)
            else:
                plt.title(f"Model: {model.split('/')[1]}, OOD data: " + ood_data)
                
            if not os.path.exists(f'plots/{model}'):
                os.makedirs(f'plots/{model}')
            plt.ylim(([0, 400]))
            plt.legend()
            plt.savefig(f'plots/{model}/{ood_data} in_out_confidence.png', dpi=300)
            plt.close()

            probs_df = pd.DataFrame(probs)
            print(probs_df.describe())
            print(len(list(filter(lambda x: x >= 1, probs_df.values.reshape(-1)))))

            print(f"labels: {labels}")
            print(f"probabilities:{probs}")

            tp, fp, tpr, fpr, fpr_at_tpr95 = get_curve(in_probs, out_probs)            

            print(f"auroc_score: {roc_auc_score(labels, probs)}")
            print(-np.trapz(1. - fpr, tpr)  * 100)
            print(f"fpr_at_tpr95: {fpr_at_tpr95}")

            # # AUIN
            # # mtype = 'AUPR-IN'
            # denom = tp['Baseline'] + fp['Baseline']
            # denom[denom == 0.] = -1.
            # pin_ind = np.concatenate([[True], denom > 0., [True]])
            # pin = np.concatenate([[.5], tp['Baseline'] / denom, [0.]])
            # auin = -np.trapz(pin[pin_ind], tpr[pin_ind])  * 100
            # print(f"AUPR-IN: {auin}")

            # # AUOUT
            # # mtype = 'AUPR-OUT'
            # denom = tp['Baseline'][0] - tp['Baseline'] + fp['Baseline'][0] - fp['Baseline']
            # denom[denom == 0.] = -1.
            # pout_ind = np.concatenate([[True], denom > 0., [True]])
            # pout = np.concatenate([[0.], (fp['Baseline'][0] - fp['Baseline']) / denom, [.5]])
            # auout = np.trapz(pout[pout_ind], 1. - fpr[pout_ind]) * 100
            
            # print(f"AUPR-OUT: {auout}")
