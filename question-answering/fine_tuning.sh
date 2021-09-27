#!/bin/sh
export SAVE_DIR=./output
export DATA_DIR=/nas/home/moo/NLU/biobert-pytorch/QA-unknown-detection/question-answering/datasets/QA/BioASQ
export TRAINDATA=BioASQ-train_split-factoid-7b.json
export INDOMAIN=BioASQ-dev-factoid-7b.json
export OUTDOMAIN=
export OFFICIAL_DIR=./scripts/bioasq_eval
export BATCH_SIZE=16
export LEARNING_RATE=5e-5
export NUM_EPOCHS=3
export MAX_LENGTH=384
export SEED=42
export CUDA_VISIBLE_DEVICES=2
export MODEL_NAME=bert-base-cased
export DOMAIN_TYPE=bioasq


# declare -a DOMAIN_PREFIXES=["squad" "music" "finance" "film" "law" "computing"]
######### model #########

# bert-base-cased
# dmis-lab/biobert-base-cased-v1.1
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
# allenai/biomed_roberta_base
# SpanBERT/spanbert-base-cased

# Train
# for domain in 0 1 2 3 4 5; do
#     # DOMAIN_PREFIXES="${DOMAIN_PREFIXES[domain]}"

python3 fine_tuning.py \
    --model_type ${MODEL_NAME} \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --train_file ${DATA_DIR}/${TRAINDATA} \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --output_dir ${SAVE_DIR}/${MODEL_NAME}/bioasq_in_domain \
    --overwrite_cache \
    --overwrite_output

# Evaluation
python3 fine_tuning.py \
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

    # /daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-dev-factoid-7b.json

# /daintlab/home/moo/NLU/biobert-pytorch/datasets/QA/BioASQ/BioASQ-test-factoid-7b_answers.json
    # /daintlab/data/NLU/QA/data/bioasq/BioASQ-7b/test/Full-Abstract/BioASQ-test-factoid-7b-3.json 
# /daintlab/data/NLU/QA/data/marco/squad.${DOMAIN_TYPE}.test.json
# /daintlab/data/NLU/squad/dev-v1.1.json
# -- train_file ${DATA_DIR}/BioASQ-train-factoid-7b.json
# --predict_file ${DATA_DIR}/BioASQ-dev-factoid-7b.json \
# --golden_file ${DATA_DIR}/7B_golden.json \
# --trian_file /daintlab/data/NLU/QA/data/marco/squad.${DOMAIN_TYPE}.train.json