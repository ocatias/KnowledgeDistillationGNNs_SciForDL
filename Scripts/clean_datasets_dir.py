"""
Cleans dataset folder by removing everything
"""

import os
import shutil

from Misc.config import config
from Scripts.clean_dir import clean

excluded_files = ["splits", "__init__.py", ".gitignore"]

def main():
    clean(config.DATA_PATH, excluded_files)

if __name__ == "__main__":
    main()