
import pdb 
import json 
from datasets import load_dataset


from trained_calibration.eval.triviaqa_evaluation import normalize_answer as tqa_normalize_answer


def get_dataset(dataset_name, limit=None, batch_size=256):
    # add a column with the input prompt 
    # we can actually use this prompt for TruthfulQA also because the questions resemble trivia questions 
    def add_trivia_qa_prompt(sample):
        sample['generator_prompt'] = f"""You will be asked trivia questions. Please respond to the best of your ability.
Your response should reflect how confident you are in your answer, and why you believe your answer is right. Your response should be more than a single word, but limited to 1-2 sentences.

Question: {sample['question']}
Response:"""
        

        # we will need to fill in the answer later on  
        sample['evaluator_prompt'] = f"""Please decide whether to accept the following response to a trivia question. Make your decision based on how confident the answer sounds.
Respond with either "yes" or "no", and nothing else. 
Question: {sample['question']}
Response:"""

        # dump to string so that we don't have issues w collation 
        if dataset_name == "trivia_qa":
            correct_answers = sample['answer']['normalized_aliases'] + [tqa_normalize_answer(ans) for ans in sample['answer'].get('human_answers', [])]
            sample['correct_answer'] = json.dumps(correct_answers)
        else:
            correct_answers = sample['correct_answers']
            incorrect_answers = sample['incorrect_answers']
            sample['correct_answer'] = json.dumps(correct_answers) 
            sample['incorrect_answer'] = json.dumps(incorrect_answers)
        return sample

    

    if dataset_name == "trivia_qa":
        dataset = load_dataset("mandarjoshi/trivia_qa", 
                           "rc.web.nocontext",
                           cache_dir="/nas-ssd2/esteng/.cache")
        dataset = dataset.map(add_trivia_qa_prompt, batched=False)
        # limit just to needed keys 
        for key in ["train", "validation", "test"]:
            columns = dataset[key].column_names
            to_remove = set(columns) - set(["generator_prompt", "evaluator_prompt", "correct_answer"])
            dataset[key] = dataset[key].remove_columns(list(to_remove)) 

    elif dataset_name == "truthful_qa": 
        dataset = load_dataset("truthful_qa", "generation",
                               cache_dir="/nas-ssd2/esteng/.cache")

        dataset = dataset.map(add_trivia_qa_prompt, batched=False)
        for key in ["validation"]:
            columns = dataset[key].column_names
            to_remove = set(columns) - set(["generator_prompt", "evaluator_prompt", "correct_answer", "incorrect_answer"])
            dataset[key] = dataset[key].remove_columns(list(to_remove)) 


    return dataset


