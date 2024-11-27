import os
import json
import glob

from Misc.config import config

def string_to_ls_of_ints(string):
    return list(map(lambda x: int(x), string.split(",")))

def find_most_recent_model(model, dataset):
    """
    Returns path of the most recently trained model and its config on the given dataset
    """
    from Misc.config import config

    possible_models = glob.glob(os.path.join(config.MODELS_PATH, "*.pt"))
    possible_models = list(filter(lambda x: dataset in x and model in x, possible_models))
    
    # Sort by recency to take most recent one
    possible_models.sort(key=lambda x: os.path.getmtime(x))
    pth_model_weights = possible_models[0]
    pth_model_config = pth_model_weights[:-3] + ".json"
    return pth_model_weights, pth_model_config    

def filter_empty_str_from_ls(ls):
    return list(filter(lambda x: x != "", ls))

def string_of_args_to_dict(str_args):    
    ls = filter_empty_str_from_ls(str_args.split("--")) 
    d = {}
    for entry in ls:
        if '=' in entry:
            split_entry = entry.split('=')
        else:
            split_entry = entry.split(' ')
            
        split_entry = filter_empty_str_from_ls(split_entry)
        assert len(split_entry) == 2
        d[f"--{split_entry[0]}"] = split_entry[1].replace(' ', '')

    return d
    
def get_basic_model_args(model, dataset):
    with open(os.path.join(config.CONFIGS_PATH, "basic_models.json"), "r") as file:
        args_dict = json.load(file)
        dataset_args = args_dict["dataset"][dataset].split("--")
        model_args = args_dict["model"][model][dataset].split("--")

    def clean_args_ls(args_ls):
        # Strip first empty string
        args_ls = args_ls[1:]
        args_ls = list(map(lambda x: (x[:-1]) if (x[-1] == " ") else x, args_ls))
        args_dict = {}
        for pair in args_ls:
            split_pair = pair.split(" ")
            args_dict[split_pair[0]] = split_pair[1]
        return  args_dict
    
    dataset_args = clean_args_ls(dataset_args)
    model_args = clean_args_ls(model_args)

    args_dict = {}
    # If an argument is specified in both the dataset and model args then use the model argument! 
    for key, value in dataset_args.items():
        args_dict[key] = value
    for key, value in model_args.items():
        args_dict[key] = value    
        
    args = ""
    for key, value in args_dict.items():
        args += f"--{key} {value} " 

    args += f" --model {model} "
    args += f"--dataset {dataset} "
    return args

def edge_tensor_to_list(edge_tensor):
    edge_list = []
    for i in range(edge_tensor.shape[1]):
        edge_list.append((int(edge_tensor[0, i]), int(edge_tensor[1, i])))

    return edge_list

def list_of_dictionary_to_dictionary_of_lists(ls):
    return {k: [dic[k] for dic in ls] for k in ls[0]}

def dictionary_of_lists_to_list_of_dictionaries(dict):
    return [dict(zip(dict,t)) for t in zip(*dict.values())]

def transform_dict_to_args_list(dictionary):
    """
    Transform dict to list of args
    """
    list_args = []
    for key,value in dictionary.items():
        list_args += [key, str(value)]
    return list_args

def classification_task(dataset_name):
    return dataset_name in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]
