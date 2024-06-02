#!/bin/bash

seed=$1
#llama_ckpt="/nas-ssd2/archiki/.cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b/"
llama_ckpt="meta-llama/Meta-Llama-3-8B" 

python trained_calibration/rl/train/train_dpo.py \
	--output_dir models/train_llama_8b_10k_balanced_long_${seed}_seed \
	--model ${llama_ckpt} \
	--reward_model mistralai/Mistral-7B-v0.1 \
	--eval_steps 40 \
	--warmup_steps 10 \
	--save_steps 40 \
	--train_dataset data/trivia_qa/tqa_10k_train.jsonl \
	--valid_dataset data/trivia_qa/tqa_full_valid.jsonl \
	--valid_limit 500 \
	--per_device_train_batch_size 6 \
	--gradient_accumulation_steps 10 \
	--per_device_eval_batch_size 2 \
	--n_eval_batches 30 \
	--max_length 200  \
	--max_steps 250 \
	--seed ${seed} \
	--balance_types true
	
