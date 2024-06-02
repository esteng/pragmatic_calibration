import csv 
import json 
import pathlib 
import argparse
import re 
import pdb 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    path = pathlib.Path("models/")
    pattern  = f"{args.pattern}/eval_data_skip.json"
    all_data = {}
    print(pattern)
    for file in path.glob(pattern): 
        with open(file) as f1:
            data = json.load(f1)
            all_data[file.parent.name] = data

    to_write = []
    for name, data in all_data.items(): 
        if "baseline" in name:
            out_row_name = "trained-baseline"
        else:
            out_row_name = "trained-main"
        if "llama" in name:
            model_name = "llama3-8b"
        else:
            model_name = "mistral-7b"

        seed = re.search("(\d+)_seed", name).group(1)

        trained_row = data['trained']
        reference_row = data['reference']
        k_number = re.search("(\d+)k", name).group(1)
        trained_row['k'] = k_number
        reference_row['k'] = k_number
        trained_row['model'] = model_name
        reference_row['model'] = model_name
        trained_row['setting'] = out_row_name
        reference_row['setting'] = "baseline"
        trained_row['seed'] = int(seed)
        reference_row['seed'] = int(seed)
        

        to_write.append(trained_row)
        if reference_row in to_write:
            continue
        to_write.append(reference_row)

    with open(args.out_file, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=to_write[0].keys())
        writer.writeheader()
        writer.writerows(to_write)

