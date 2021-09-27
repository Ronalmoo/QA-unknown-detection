import json
import csv
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from datasets import load_dataset
import spacy
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer


from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    get_cosine_schedule_with_warmup,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)



def spacy_tokenizer(document):
    nlp = spacy.load("en_core_web_sm", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    # tokenize the document with spaCY
    
    doc = nlp(document)
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for token in doc if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.text.strip() != '' and \
        token.text.find("\n") == -1)]
    return tokens

def dfreq(idf, N):
    return (1 + N) / np.exp(idf - 1) - 1


def get_new_vocab(og_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(og_tokenizer)
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=None, 
                                   norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    
    files = "/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-train_split-factoid-7b.json"
    with open(files) as f:
        file = json.load(f)
    contexts = []
    doc1 = ""
    doc2 = ""
    cnt = 0
    for question in file['data']:
        for paragraph in question['paragraphs']:
            # contexts.append(paragraph['context'])
            # if cnt < 1650:
            doc1 += paragraph['context']
            # else:
                # doc2 += paragraph['context']
            # cnt += 1
    contexts.append(doc1)
    # contexts.append(doc2)
    docs = contexts
    length = len(docs)
    result = tfidf_vectorizer.fit_transform(docs)
    idf = tfidf_vectorizer.idf_

    # sorted idf, tokens and docs frequencies
    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    idf_sorted = idf[idf_sorted_indexes]
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names())[idf_sorted_indexes]
    dfreqs_sorted = dfreq(idf_sorted, length).astype(np.int32)
    tokens_dfreqs = {tok:dfreq for tok, dfreq in zip(tokens_by_df,dfreqs_sorted)}
    tokens_pct_list = [int(round(dfreq/length*100,2)) for token,dfreq in tokens_dfreqs.items()]


    number_tokens_with_DF_above_pct = list()
    for pct in range(1,101,50):
        index_max = len(np.array(tokens_pct_list)[np.array(tokens_pct_list)>=pct])
        number_tokens_with_DF_above_pct.append(index_max)

    df_docfreqs = pd.DataFrame({'pct':list(range(1,101,50)),'number of tokens with DF above pct%':number_tokens_with_DF_above_pct})

    pct = 1
    index_max = len(np.array(tokens_pct_list)[np.array(tokens_pct_list)>=pct])
    new_tokens = tokens_by_df[:index_max]
    # print(len(new_tokens))

    old_vocab = [k for k,v in tokenizer.get_vocab().items()]
    new_vocab = [token for token in new_tokens]
    idx_old_vocab_list = list()
    same_tokens_list = list()
    different_tokens_list = list()

    for idx_new,w in enumerate(new_vocab): 
        try:
            idx_old = old_vocab.index(w)
        except:
            idx_old = -1
        if idx_old >= 0:
            idx_old_vocab_list.append(idx_old)
            same_tokens_list.append((w,idx_new))
        else:
            different_tokens_list.append((w,idx_new))

    new_tokens = [k for k,v in different_tokens_list] 

    # resize the embeddings matrix of the model 
    # model.resize_token_embeddings(len(tokenizer))
    return new_tokens


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=spacy_tokenizer, 
                                   norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    
    files = "/daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-train_split-factoid-7b.json"
    with open(files) as f:
        file = json.load(f)
    contexts = []
    doc1 = ""
    doc2 = ""
    cnt = 0
    for question in file['data']:
        for paragraph in question['paragraphs']:
            # contexts.append(paragraph['context'])
            if cnt < 1700:
                doc1 += paragraph['context']
            else:
                doc2 += paragraph['context']
            cnt += 1
    contexts.append(doc1)
    # contexts.append(doc2)

    # contexts = []
    # doc1, doc2 = "", ""
    # dataset1 = load_dataset("scientific_papers", "pubmed")
    # dataset2 = load_dataset("scientific_papers", "arxiv")
    
    # for i in dataset1['train']['article'][:10]:
    #     doc1 += i
    # for i in dataset2['train']['article'][:10]:
    #     doc2 += i

    # contexts.append(doc1)
    # contexts.append(doc2)

    # print(contexts)
    docs = contexts
    length = len(docs)
    print(length)
    
    result = tfidf_vectorizer.fit_transform(docs)

    # df = pd.DataFrame(result[0].T.todense(),
        #  index = tfidf_vectorizer.get_feature_names(), columns=["TF-IDF"])
    # df = df.sort_values('TF-IDF', ascending=False)
    # print(df.columns)
    # import sys; sys.exit()
    
    idf = tfidf_vectorizer.idf_

    # sorted idf, tokens and docs frequencies
    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    idf_sorted = idf[idf_sorted_indexes]
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names())[idf_sorted_indexes]
    dfreqs_sorted = dfreq(idf_sorted, length).astype(np.int32)
    tokens_dfreqs = {tok:dfreq for tok, dfreq in zip(tokens_by_df,dfreqs_sorted)}
    tokens_pct_list = [int(round(dfreq/length*100,2)) for token,dfreq in tokens_dfreqs.items()]


    number_tokens_with_DF_above_pct = list()
    for pct in range(1,101,50):
        index_max = len(np.array(tokens_pct_list)[np.array(tokens_pct_list)>=pct])
        number_tokens_with_DF_above_pct.append(index_max)

    df_docfreqs = pd.DataFrame({'pct':list(range(1,101,50)),'number of tokens with DF above pct%':number_tokens_with_DF_above_pct})
    print(df_docfreqs.transpose())
    

    pct = 1
    index_max = len(np.array(tokens_pct_list)[np.array(tokens_pct_list)>=pct])
    new_tokens = tokens_by_df[:index_max]
    # print(len(new_tokens))

    old_vocab = [k for k,v in tokenizer.get_vocab().items()]
    new_vocab = [token for token in new_tokens]
    idx_old_vocab_list = list()
    same_tokens_list = list()
    different_tokens_list = list()

    for idx_new,w in enumerate(new_vocab): 
        try:
            idx_old = old_vocab.index(w)
        except:
            idx_old = -1
        if idx_old>=0:
            idx_old_vocab_list.append(idx_old)
            same_tokens_list.append((w,idx_new))
        else:
            different_tokens_list.append((w,idx_new))


    print(len(same_tokens_list))
    print(len(different_tokens_list))
    print(len(same_tokens_list)+len(different_tokens_list))

    new_tokens = [k for k,v in different_tokens_list]
    # print(len(new_tokens), new_tokens[:20])

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model_name = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
    # import pdb; pdb.set_trace()
    added_tokens = tokenizer.add_tokens(new_tokens)

    print("[ AFTER ] tokenizer vocab size:", len(tokenizer)) 
    print()
    print('added_tokens:', added_tokens)
    print()
    # resize the embeddings matrix of the model 
    model.resize_token_embeddings(len(tokenizer))
    print(model)
    print(new_tokens[500:-100])