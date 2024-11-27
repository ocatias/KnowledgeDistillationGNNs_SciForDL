"""
Script to quickly train (and possibly store) a model based on predefined hyperparameters 
(see: Configs/basic_models.json)
"""

import argparse

from Misc.utils import get_basic_model_args, string_of_args_to_dict
from Exp.run_model import run as train_model

def main():
    parser = argparse.ArgumentParser(description='Quickly train a model based on hyperparameters specified under Configs.')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('--store_model', type=int, default=1)   
    
    args = parser.parse_args()
    model_args = get_basic_model_args(args.model, args.dataset)
    
    params_dict =  string_of_args_to_dict(model_args)
    params_dict["--store_model"] = args.store_model
    
    train_model(params_dict)
    
if __name__ == "__main__":
    main()