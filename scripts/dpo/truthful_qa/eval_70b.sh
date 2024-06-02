
#!/bin/bash


SEED=$2

python trained_calibration/rl/evaluate_large.py \
    --model "mistralai/Mistral-7B-v0.1" \
    --trained_model $1 \
    --data_path "data/truthful_qa/truthful_qa_test_mistral_generator_mistral_evaluator.jsonl" \
    --limit -1 \
    --out_path $(dirname ${1})/truthful_qa_eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed ${SEED} \
    --threshold 0.66
