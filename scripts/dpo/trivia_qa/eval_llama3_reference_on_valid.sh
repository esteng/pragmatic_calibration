#!/bin/bash

# script to get reference numbers from llama3 

SEED=$1

llama_ckpt="/nas-ssd2/archiki/.cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b/"
mkdir -p models/llama3-8b_reference_${SEED}_seed/
python trained_calibration/rl/evaluate_dpo.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --trained_model  ${llama_ckpt}  \
    --data_path "data/trivia_qa/tqa_validation_mistral_v1_small.jsonl" \
    --limit -1 \
    --out_path models/llama3-8b_reference_${SEED}_seed/eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed ${SEED} \
    --threshold 0.66
