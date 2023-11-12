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

def fetch_tickers() -> List[str]:
    """fetch tickers with qualified SEC data

    Returns:
        List[str]: _description_
    """
    tickers = []
    for fpath in filter_files(10):
        with open (fpath, 'r', encoding='utf8') as f:
            ticker = re.search(r'\/([a-z-]+).txt', fpath).group(1)
            tickers.append(ticker.replace('-', '.'))

    return tickers

def make_gics_look_up(
    gics_type: str = 'industry'
) -> Tuple[Dict[str, str], Dict[int, str]]:
    """make ticker to GICS industry lookup dict

    Args:
        gics_type (str, optional): gics granularity. Defaults to 'industry'.

    Returns:
        Tuple[Dict[str, str], Dict[int, str]]: ticker as key and GICS industry as value
    """
    if gics_type == 'industry':
        digit_pos = 6
    elif gics_type == 'industry_group':
        digit_pos = 4
    elif gics_type == 'sector':
        digit_pos = 2

    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(d, 'data', 'raw_data')
    tickers = fetch_tickers()

    gics_df = pd.read_csv(
        os.path.join(data_dir, 'ticker_to_gics.csv'),
        names=['ticker', 'gics'],
        dtype={'ticker': str, 'gics': str}
        )
    gics_df['ticker'] = gics_df['ticker'].str.lower().str.replace('/', '.')
    gics_df = gics_df[gics_df['ticker'].isin(tickers)]

    gics_df['industry'] = gics_df['gics'].apply(lambda x: x[:digit_pos])
    int_mapping = dict(enumerate(set(gics_df['industry'])))
    int_mapping = {v: k for k, v in int_mapping.items()}
    gics_df['industry'] = gics_df['industry'].replace(int_mapping)

    # make industry
    gics_lookup = dict(zip(gics_df['ticker'].str.lower(), gics_df['industry']))
    # for downstream multiple gics
    reverse_int_mapping = {v: k for k, v in int_mapping.items()}

    return gics_lookup, reverse_int_mapping

def make_input_data(
    gics_type: str = 'industry'
) -> Tuple[List[str], List[str], List[str]]:
    """make ompany business description from files

    Returns:
        Tuple[List[str], List[str], List[str]]: list text, list of label, and tickers
    """
    gics_mapping, _ = make_gics_look_up(gics_type=gics_type)
    texts = []
    gics = []
    tickers = []
    for fpath in filter_files(10):
        with open (fpath, 'r', encoding='utf8') as f:
            texts.append(f.read().replace('\n', ''))
            ticker = re.search(r'\/([a-z-]+).txt', fpath).group(1)
            gics.append(gics_mapping[ticker.replace('-', '.')])
            tickers.append(ticker)

    assert len(texts) == len(gics), 'text and ticker length are inconsistent'

    return texts, gics, tickers


def gics_code_to_string(gics_type: str = 'industry') -> Dict[str, str]:
    """make indsutry code to industry string mapping

    Args:
        gics_type (str, optional): gics granularity. Defaults to 'industry'.

    Returns:
        Dict[str, str]: mapping dict
    """
    if gics_type == 'industry':
        gics_col = 'industry'
        gics_code_col = 'industry_code'
    elif gics_type == 'industry_group':
        gics_col = 'industry_group'
        gics_code_col = 'industry_group_code'
    elif gics_type == 'sector':
        gics_col = 'sector'
        gics_code_col = 'sector_code'

    d = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(d, 'data', 'raw_data')
    gics_mapping_df = pd.read_csv(os.path.join(data_dir, 'gics_mapping.csv'))
    industry_code = gics_mapping_df[gics_code_col].astype(str)
    industry = gics_mapping_df[gics_col]

    return dict(zip(industry_code, industry))


if __name__ == '__main__':
    input_text, labels, tickers = make_input_data()
