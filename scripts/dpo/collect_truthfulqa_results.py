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
    trained_pattern  = f"{args.pattern}/truthful_qa_trained.json"
    reference_pattern  = f"{args.pattern}/truthful_qa_reference.json"
    all_trained_data = {}
    all_reference_data = {}

    for file in path.glob(trained_pattern): 
        with open(file) as f1:
            data = json.load(f1)
            all_trained_data[file.parent.name] = data

    for file in path.glob(reference_pattern):
        with open(file) as f1:
            data = json.load(f1)
            all_reference_data[file.parent.name] = data

    to_write = []
    for dtype, data_dict  in zip(["trained", "reference"], [all_trained_data, all_reference_data]):
        for name, data in data_dict.items(): 
            if dtype == "trained":
                if "baseline" in name:
                    out_row_name = "trained-baseline"
                else:
                    out_row_name = "trained-main"
            else:
                out_row_name = "baseline"
            if "llama" in name:
                model_name = "llama3-8b"
            else:
                model_name = "mistral-7b"

            seed = re.search("(\d+)_seed", name).group(1)

            truthfulness = data['truth']
            informative = data['info']

            to_write.append({"setting": out_row_name, "model": model_name, "seed": int(seed), "truthfulness": truthfulness, "informative": informative})
            


    with open(args.out_file, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=to_write[0].keys())
        writer.writeheader()
        writer.writerows(to_write)

