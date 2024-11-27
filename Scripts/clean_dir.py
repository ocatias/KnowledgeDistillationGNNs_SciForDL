
import os
import shutil

from Misc.config import config

def clean(path_dir, excluded_files = []):
    """
    Remove every file in path_dir except excluded_files
    """
    
    # https://stackoverflow.com/a/185941
    for filename in os.listdir(path_dir):
        if filename in excluded_files:
            continue
        file_path = os.path.join(path_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))