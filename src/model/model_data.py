import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from src.meta_data import get_meta_data


def filter_files(min_kb: int = 10) -> List[str]:
    """fetch tickers size greter than a threshold

    Args:
        min_kb (int, optional): size in kb. Defaults to 10.
    """
    save_dir = save_dir = get_meta_data()['SEC_DIR']
    file_names = os.listdir(save_dir)
    file_size = [os.path.getsize(os.path.join(save_dir, f)) / 1024 for f in file_names]
    file_to_use = [os.path.join(save_dir, f) for f, s in zip(file_names, file_size) if s >= min_kb]

    return file_to_use

# model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
def make_gics_look_up() -> Dict[str, str]:
    """make ticker to GICS industry lookup dict

    Returns:
        Dict[str, str]: ticker as key and GICS industry as value
    """
    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(d, 'data', 'raw_data')

    gics_df = pd.read_csv(
        os.path.join(data_dir, 'ticker_to_gics.csv'),
        names=['ticker', 'gics'],
        dtype={'ticker': str, 'gics': str}
        )

    gics_df['industry'] = gics_df['gics'].apply(lambda x: x[:6])
    int_mapping = dict(enumerate(set(gics_df['industry'])))
    int_mapping = {v: k for k, v in int_mapping.items()}
    gics_df['industry'] = gics_df['industry'].replace(int_mapping)

    # make industry
    gics_lookup = dict(zip(gics_df['ticker'].str.lower(), gics_df['industry']))

    return gics_lookup

def make_input_data() -> Tuple[List[str], List[str]]:
    """make ompany business description from files

    Returns:
        Tuple[List[str], List[str]]: list text and list of label
    """
    gics_mapping = make_gics_look_up()
    texts = []
    gics = []
    for fpath in filter_files(10):
        with open (fpath, 'r', encoding='utf8') as f:
            texts.append(f.read().replace('\n', ''))
            ticker = re.search(r'[^a-z]([a-z]+).txt', fpath).group(1)
            gics.append(gics_mapping[ticker])

    assert len(texts) == len(gics), 'text and ticker length are inconsistent'

    return texts, gics


if __name__ == '__main__':
    input_text, labels = make_input_data()
