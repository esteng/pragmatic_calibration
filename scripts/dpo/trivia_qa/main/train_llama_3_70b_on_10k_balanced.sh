#!/bin/bash

seed=$1
llama_ckpt="meta-llama/Meta-Llama-3-70B"
python trained_calibration/rl/train/train_dpo.py \
	--output_dir models/train_llama_70b_10k_balanced_long_fast_${seed}_seed \
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
	--per_device_eval_batch_size 3 \
	--n_eval_batches 20 \
	--max_length 200  \
	--max_steps 250 \
	--seed ${seed} \
	--balance_types true
	
