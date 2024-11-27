"""
Removes trained models, created embeddings and datasets
"""

import os
import shutil

from Misc.config import config
from Scripts.clean_dir import clean
from Scripts.clean_datasets_dir import main as clean_datasets

def main():
    # Datasets
    clean_datasets()
    
    # Embs
    clean(config.EMBS_PATH)
    
    # Trained models
    clean(config.MODELS_PATH)

if __name__ == "__main__":
    main()