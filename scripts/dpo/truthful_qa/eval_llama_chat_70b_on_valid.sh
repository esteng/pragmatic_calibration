#!/bin/bash

SEED=$1

mkdir -p models/llama3-70b_instruct_${SEED}_seed/
python trained_calibration/rl/evaluate_with_chat_llama.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --trained_model  "meta-llama/Meta-Llama-3-70B-Instruct" \
    --data_path "data/truthful_qa/truthful_qa_test_mistral_generator_mistral_evaluator.jsonl" \
    --limit -1 \
    --out_path models/llama3-70b_instruct_${SEED}_seed/truthful_qa_eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed ${SEED} \
    --threshold 0.66
