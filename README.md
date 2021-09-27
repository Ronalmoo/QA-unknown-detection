# Unknown detection in QA systems
This repository provides the code for unknown detection(including fine tuning models)

## Download
I used the model on the huggingface.

* **[BERT-base-cased](https://huggingface.co/bert-base-cased)**
* **[Biobert-base-cased-v1.1 Copied](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)**
* **[PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)**
* **[BioMedRoBERTa](https://huggingface.co/allenai/biomed_roberta_base)**
* **[SpanBERT-base-cased](https://huggingface.co/SpanBERT/spanbert-base-cased)**

## Installation
Install requirements
```bash
$ pip install -r requirements.txt
```

## Datasets

We provide a pre-processed version of benchmark datasets for each task as follows:
* In domain
*   **[`BioASQ`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**
* Out domain
*   **[`SQuAD 1.1`](https://huggingface.co/datasets/squad)**
*   **[`MS MARCO`](https://drive.google.com/drive/folders/1_Av713NR_3uqzU4au1xJpKB-h5QNxsep?usp=sharing)**: 5 domains(music, computing, law, film, finance) on question answering task.

## Fine-tuning

```bash
SAVE_DIR=./output
DATA_DIR=''
TRAINDATA=BioASQ-train_split-factoid-7b.json
INDOMAIN=BioASQ-dev-factoid-7b.json
OUTDOMAIN=''
OFFICIAL_DIR=./scripts/bioasq_eval
BATCH_SIZE=16
LEARNING_RATE=5e-5
NUM_EPOCHS=3
MAX_LENGTH=384
SEED=42
CUDA_VISIBLE_DEVICES=2
MODEL_NAME=bert-base-cased
DOMAIN_TYPE=bioasq

# Finetuning with indomain data
$ python3 fine_tuning.py \
    --model_type ${MODEL_NAME} \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --train_file ${DATA_DIR}/${INDOMAIN} \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --output_dir ${SAVE_DIR}/${MODEL_NAME}/bioasq_in_domain \
    --overwrite_cache \
    --overwrite_output 

# Inference with indomain or outdomain data
$ python3 fine_tuning.py \
    --model_type ${MODEL_NAME} \
    --model_name_or_path ${SAVE_DIR}/${MODEL_NAME}/bioasq_in_domain \
    --do_eval \
    --predict_file ${DATA_DIR}/${TESTDATA} \
    --golden_file ${DATA_DIR}/7B_golden.json \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --official_eval_dir ${OFFICIAL_DIR} \
    --output_dir ${SAVE_DIR}/${MODEL_NAME}/${DOMAIN_TYPE}_bioasq_dev \
    --eval_all_checkpoints \
    --overwrite_cache \

or

$ sh fine_tuning.sh

```

## Unknown detection
* Get AUROC
```bash
python3 get_auroc.py --path 'your path that contains n_best_prediction.json '

```
* Get AURC 
```bash
python3 get_aurc.py --path 'your path that contains n_best_prediction.json '
```


## Results

* Indomain generalization
<img width="642" alt="generalization" src="https://user-images.githubusercontent.com/44221520/124387986-921a8880-dd1b-11eb-98f8-f61d07806117.png">

* Out of domain detection(AUROC)

<img width="635" alt="auroc" src="https://user-images.githubusercontent.com/44221520/124387758-af9b2280-dd1a-11eb-99ab-285c827593f7.png">

* Out of domain detection(AURC)
<img width="633" alt="AURC" src="https://user-images.githubusercontent.com/44221520/124387990-95157900-dd1b-11eb-845d-ca47d85d3615.png">

## Contact Information
For help or issues, please submit a GitHub issue. 
(`ydaniel0826 (at) gmail.com`)
