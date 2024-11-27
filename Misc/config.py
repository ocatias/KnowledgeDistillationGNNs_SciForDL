import yaml

class dotdict(dict):
    """dot.notation access to dictionary attributes
    Source: https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict(yaml.safe_load(open("./Configs/config.yaml")))

