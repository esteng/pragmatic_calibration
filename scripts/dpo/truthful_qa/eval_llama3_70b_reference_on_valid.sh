#!/bin/bash

# script to get reference numbers from llama3 

SEED=$1

llama_ckpt="meta-llama/Meta-Llama-3-70B"
mkdir -p models/llama3-70b_reference_${SEED}_seed/
python trained_calibration/rl/evaluate_large.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --trained_model  ${llama_ckpt}  \
    --data_path "data/truthful_qa/truthful_qa_test_mistral_generator_mistral_evaluator.jsonl" \
    --limit -1 \
    --out_path models/llama3-70b_reference_${SEED}_seed/truthful_qa_eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed ${SEED} \
    --threshold 0.66
