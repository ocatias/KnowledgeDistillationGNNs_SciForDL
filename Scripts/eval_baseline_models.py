import os
import json
import argparse

from Exp.run_experiment import main as run_experiment

from Misc.utils import get_basic_model_args
from Misc.config import config

def main():
    parser = argparse.ArgumentParser(description='Quickly train a model based on hyperparameters specified under Configs.')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('--repeats', type=int, default=3,
                    help="Number of times to repeat the final model training")
    args = parser.parse_args()
    
    generated_configs_pth = os.path.join(config.CONFIGS_PATH, "Auto")
    if not os.path.isdir(generated_configs_pth):
        os.mkdir(generated_configs_pth)
    
    model_args = get_basic_model_args(args.model, args.dataset)
    
    # Create config
    file_path = os.path.join(generated_configs_pth, f"{args.dataset}_{args.model}.yaml")
    with open(file_path, "w") as file:
        for pair in model_args.split("--")[1:]:
            split_pair = pair.split(" ")
            key = split_pair[0]
            value = split_pair[1]
            if value not in ["False", "True"]:
                try:
                    float(value)
                except ValueError:
                    value = "\"" + value + "\""
            
            file.write(f"{key}: [{value}]\n")
    
    exp_args = {
        "-grid": file_path,
        "-dataset": args.dataset,
        "--repeats": args.repeats
    }
    run_experiment(exp_args)
            
if __name__ == "__main__":
    main()