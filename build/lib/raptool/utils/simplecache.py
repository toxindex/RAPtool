# simplecache.py

import os
import hashlib
import pickle
import functools
from pathlib import Path

def file_content_hash(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def is_file_path(val):
    # Accepts str or Path, and checks if it points to a file
    if isinstance(val, (str, Path)) and os.path.isfile(val):
        return True
    return False

def simple_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Replace all file path args/kwargs with their content hash
            args_list = []
            for arg in args:
                if is_file_path(arg):
                    args_list.append(f"filehash:{file_content_hash(arg)}")
                else:
                    args_list.append(arg)
            new_kwargs = {}
            for k, v in kwargs.items():
                if is_file_path(v):
                    new_kwargs[k] = f"filehash:{file_content_hash(v)}"
                else:
                    new_kwargs[k] = v
            key = f"{func.__name__}:{args_list}:{new_kwargs}"
            hash_key = hashlib.sha256(key.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{hash_key}.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result

        return wrapper

    return decorator
