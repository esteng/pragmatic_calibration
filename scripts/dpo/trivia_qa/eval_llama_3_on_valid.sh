
#!/bin/bash

llama_ckpt="/nas-ssd2/archiki/.cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b/"
python trained_calibration/rl/evaluate_dpo.py \
    --model ${llama_ckpt} \
    --trained_model $1 \
    --data_path "data/trivia_qa/tqa_validation_mistral_v1_small.jsonl" \
    --limit -1 \
    --out_path $(dirname ${1})/eval_dpo_on_valid.jsonl \
    --n_per_prompt 1 \
    --seed 12 \
    --threshold 0.66
