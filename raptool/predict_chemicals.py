import pandas as pd
import numpy as np
from pathlib import Path
import raptool.utils.chemprop as chemprop
from typing import List, Dict, Any
import raptool.utils.simplecache as simplecache
import logging
import time 
import itertools
from tqdm import tqdm
import pickle

def predict_chemicals(input_path, output_path):
    """
    Predict chemical properties using chemprop and save results.
    
    Args:
        input_path (str or pathlib.Path): Path to the input CSV file containing chemicals with 'inchi' column
        output_path (str or pathlib.Path): Path to save the output parquet file with predictions
    """
    # Convert paths to pathlib.Path objects if they're not already
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Read input data
    indf = pd.read_csv(input_path)
    
    # Setup cache for predictions
    cachedir = Path('cache') / 'function_cache' / 'chemprop_predictions_cache'
    cachedir.mkdir(parents=True, exist_ok=True)
    pred = simplecache.simple_cache(cachedir)(chemprop.chemprop_predict_all)

    # Setup failed chemicals cache (now a dict: inchi -> fail count)
    failed_cache_path = Path('cache') / 'failed_chemicals.pkl'
    failed_chemicals = dict()
    if failed_cache_path.exists():
        try:
            with open(failed_cache_path, 'rb') as f:
                failed_chemicals = pickle.load(f)
            logging.info(f"Loaded {len(failed_chemicals)} failed chemicals from cache")
        except Exception as e:
            logging.warning(f"Could not load failed chemicals cache: {e}")

    def safe_predict(inchi):
        # Only skip if failed more than once
        fail_count = failed_chemicals.get(inchi, 0)
        if fail_count >= 2:
            logging.info(f"Skipping chemical (failed {fail_count} times): {inchi[:50]}...")
            return []
        try:
            return pred(inchi) if inchi else None
        except Exception as e:
            logging.error(f"Error predicting for inchi {inchi}: {e}")
            # Increment fail count
            failed_chemicals[inchi] = fail_count + 1
            return []
    
    # Make predictions
    predictions = [safe_predict(inchi) for inchi in tqdm(indf['inchi'])]
    predictions = list(itertools.chain.from_iterable(predictions))
    pdf = pd.DataFrame(predictions)
    
    # Save failed chemicals cache
    try:
        with open(failed_cache_path, 'wb') as f:
            pickle.dump(failed_chemicals, f)
        logging.info(f"Saved {len(failed_chemicals)} failed chemicals to cache")
    except Exception as e:
        logging.warning(f"Could not save failed chemicals cache: {e}")
    
    # Extract property info
    pdf['property_title'] = pdf['property'].apply(lambda x: str(x.get('title', ''))).astype(str)
    pdf['property_source'] = pdf['property'].apply(lambda x: str(x.get('source', ''))).astype(str)
    pdf['property_categories'] = pdf['property'].apply(lambda x: str(x.get('categories', ''))).astype(str)
    pdf['property_metadata'] = pdf['property'].apply(lambda x: str(x.get('metadata', ''))).astype(str)
    pdf['property'] = pdf['property'].astype(str)
    
    # Merge with categorization data
    resdf = pd.merge(pdf, indf, on='inchi', how='left')
    resdf.to_parquet(output_path)
    
    logging.info(f"Saved results to {output_path}")
    
    return resdf
