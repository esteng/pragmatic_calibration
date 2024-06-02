#!/bin/bash

SEED=$1

mistral_ckpt="mistralai/Mistral-7B-v0.1" 

python trained_calibration/rl/train/train_dpo.py \
	--output_dir models/trivia_qa_8k_real_${SEED}_seed \
	--model ${mistral_ckpt} \
	--reward_model ${mistral_ckpt} \
	--eval_steps 40 \
	--warmup_steps 40 \
	--save_steps 40 \
	--train_dataset data/trivia_qa/tqa_8k_train.jsonl \
	--valid_dataset data/trivia_qa/tqa_full_valid.jsonl \
	--valid_limit 500 \
	--per_device_train_batch_size 6 \
	--gradient_accumulation_steps 10 \
	--per_device_eval_batch_size 2 \
	--n_eval_batches 30 \
	--max_length 200  \
	--max_steps 200 \
	--seed ${SEED} \
	--balance_types true
	
