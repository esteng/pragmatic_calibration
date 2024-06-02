import jsonargparse
from tqdm import tqdm 
import torch
import json 
from collections import defaultdict
import pdb 
import re

from trained_calibration.rl.dataset.dataset import get_dataset
from trained_calibration.rl.reward_model import RewardModel
from trained_calibration.rl.dataset.postprocess import postprocess_answers, postprocess_extract

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main(args):

    if args.split is None:
        split = "train"
    else:
        split = args.split


    dataset = get_dataset(args.dataset)
    if args.limit is not None:
        dataset_to_run = dataset[split].select(range(args.limit))
    else: 
        dataset_to_run = dataset[split]

    dataloader = torch.utils.data.DataLoader(
            dataset_to_run,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    
    generator = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir="/nas-ssd2/esteng/.cache").half()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/nas-ssd2/esteng/.cache") 
    device = args.model_device_map['main']
    # device = f"cuda:{args.model_device_list}"

    pad_token = None
    if "llama" in args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_token = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        pad_token = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    generator.to(device)

    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )
    # reward_model_names = args.reward_model_names.split(",")
    # reward_model_devices = args.reward_model_devices.split(",")

    reward_models = [RewardModel(model_name, args.model_device_map[f'reward{i}'], quantization_config=bnb_config) for i, model_name in enumerate(args.reward_model_names)]

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": pad_token,
    }

    with open(args.output_dir, "w") as f1:
        for batch in tqdm(dataloader): 
            try:

                prompts = batch["generator_prompt"]
                query_tensors = tokenizer(prompts, padding="longest", truncation=True, 
                                    return_tensors="pt").input_ids.to(generator.device)

                responses_by_example = defaultdict(list)
                responses_by_example_final = defaultdict(list)
                for i in range(args.n_generations):
                    response_tensors = generator.generate(query_tensors, **generation_kwargs) 
                    batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

                    batch_responses_clean, batch_answers, batch_rationales = postprocess_extract(prompts, batch_responses, generator, tokenizer, args.dataset)

                    for query_ids in range(len(query_tensors)):
                        prompt = prompts[query_ids]
                        responses_by_example[query_ids].append({"prompt": prompt, 
                                                                "response_clean": batch_responses_clean[query_ids], 
                                                                "response_orig": batch_responses[query_ids],
                                                                "answer": batch_answers[query_ids]})

                batch_questions = batch['evaluator_prompt']
                batch_correct_answers = [json.loads(x) for x in batch['correct_answer']]
                for query_idx, responses in responses_by_example.items():
                    question = batch_questions[query_idx]
                    correct_answers = batch_correct_answers[query_idx]
                    correct_answers = [correct_answers for _ in range(len(responses))]
                    question_batch = [question for _ in range(len(responses))]  
                    response_batch = [r["response_clean"] for r in responses]
                    answer_batch = [r["answer"] for r in responses]
                    # rationale_batch = [r["rationale"] for r in responses]

                    all_probs = []
                    for reward_model in reward_models:
                        if len(question_batch) > args.batch_size:
                            # chunk up
                            rewards, corrects, probs = [], [], []
                            for i in range(0, len(question_batch), args.batch_size):
                                rs, cs, ps, = reward_model(question_batch[i:i+args.batch_size], 
                                                                        response_batch[i:i+args.batch_size], 
                                                                        answer_batch[i:i+args.batch_size], 
                                                                        correct_answers[i:i+args.batch_size]) 
                                rewards.extend(rs)
                                corrects.extend(cs)
                                probs.extend(ps)

                        else:
                            rewards, corrects, probs, = reward_model(question_batch, response_batch, answer_batch, correct_answers) 
                        # vote over reward model probs
                        probs = [x.detach().cpu().item() for x in probs]
                        all_probs.append(probs)
                    all_probs = np.array(all_probs)
                    mean_probs = np.mean(all_probs, axis=0)

                    for response, mean_prob, all_prob, correct, correct_answer in zip(responses, mean_probs, all_probs.T, corrects, correct_answers):
                        response['all_probs'] = all_prob.tolist()
                        response['query_idx'] = query_idx 
                        response["mean_prob"] = mean_prob
                        response["correct"] = correct
                        response["correct_answers"] = json.dumps(correct_answer)
                        responses_by_example_final[query_idx].append(response)

                        f1.write(json.dumps(response) + "\n")
            except RuntimeError:
                print(f"Batch OOM, skipping")
                continue 

def extract_response(response):
    try:
        prompt, response = re.split("([Aa]nswer:)", response)
    except ValueError:
        return None
    # TODO (elias): remove incomplete sentences 
    return response.strip()

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", action=jsonargparse.ActionConfigFile, help="path to config file")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_names", type=list, default=None, help="list of reward model names") 
    parser.add_argument("--model_device_map", type=dict, default="0", help="dict specifying which devices have which model")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=6)

    args = parser.parse_args()

    main(args)