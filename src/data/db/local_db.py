import os
import re
import sqlite3
from importlib.resources import files
from sqlite3 import Connection
from typing import Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.db.queries import (CREATE_BUSINESS_DESC_DEF,
                                 CREATE_GICS_MAPPING_DEF, CREATE_GICS_PROB_DEF)
from src.meta_data import get_meta_data
from src.model.embedding import bert_embedding, sbert_embedding
from src.model.model_data import filter_files, make_input_data
from src.model.mutliple_gics import top_classes


def create_embedding_table_def(tbl_name: str, length: int) -> str:
    """create local embedding table

    Args:
        tbl_name (str): table name
        length (int): embedding length

    Returns:
        str: table def
    """
    col_def = ''
    for i in range(length):
        col_def += f'c_{i} FLOAT NOT NULL, \n'

    first_line = f'CREATE TABLE IF NOT EXISTS {tbl_name} (\n'
    last_line = f'ticker TEXT NOT NULL);'

    return first_line + col_def + last_line


def create_prob_table_def(tbl_name: str) -> str:
    """query to create prob table

    Args:
        tbl_name (str): talbe name

    Returns:
        str: table def
    """
    return CREATE_GICS_PROB_DEF.format(tbl_name)

def create_gics_mapping(
    key_series: pd.Series,
    value_series:pd.Series
) -> Dict[str, str]:
    return {k: v for k, v in zip(key_series, value_series)}

def upload_ticker_gics_mapping() -> str:
    fpath = os.path.join(files('src'), 'data/raw_data/ticker_to_gics.csv')
    gics_df = pd.read_csv(fpath, names=['ticker', 'gics'], dtype=str)

    fpath = os.path.join(files('src'), 'data/raw_data/gics_mapping.csv')
    mapping_df = pd.read_csv(fpath, dtype=str)

    subindustry_mapping = create_gics_mapping(
        mapping_df['subindustry_code'], mapping_df['subindustry']
    )

    industry_mapping = create_gics_mapping(
        mapping_df['subindustry_code'], mapping_df['industry']
    )

    industry_group_mapping = create_gics_mapping(
        mapping_df['subindustry_code'], mapping_df['industry_group']
    )

    sector_mapping = create_gics_mapping(
        mapping_df['subindustry_code'], mapping_df['sector']
    )
    gics_df['sector'] = gics_df['gics'].map(sector_mapping)
    gics_df['industry_group'] = gics_df['gics'].map(industry_group_mapping)
    gics_df['industry'] = gics_df['gics'].map(industry_mapping)
    gics_df['sub_industry'] = gics_df['gics'].map(subindustry_mapping)
    gics_df['ticker'] = gics_df['ticker'].str.lower()
    del gics_df['gics']

    conn.execute(CREATE_GICS_MAPPING_DEF)
    gics_df.to_sql('gics_mapping', conn, index=False, if_exists='replace')


def upload_bert_embedding(conn: Connection, num_seg: int) -> None:
    """upload bert embeddings to local db

    Args:
        conn (sqlite3.Connection): local db connection
        num_seg (int): number of text segments
    """
    texts, _, tickers = make_input_data()
    embeddings = bert_embedding(
        model_name=get_meta_data()['BERT_MODEL_DIR'],
        texts=texts,
        chunk_length=512,
        num_seg=3
    )
    cols = [f'c_{i}' for i in range(768)]
    embedding_df = pd.DataFrame(embeddings, columns=cols)
    embedding_df['ticker'] = tickers

    tbl_name = f'bert_embedding_{512 * num_seg}'
    embedding_df.to_sql(tbl_name, conn, index=False, if_exists='replace')


def upload_business_desc(conn: Connection) -> None:
    business_desc = []
    tickers = []
    for fpath in filter_files(10):
        with open (fpath, 'r', encoding='utf8') as f:
            business_desc.append(f.read().replace('\n', ''))
            ticker = re.search(r'\/([a-z-]+).txt', fpath).group(1)
            tickers.append(ticker)

    desc_df = pd.DataFrame({'ticker': tickers, 'business': business_desc})
    desc_df.to_sql('business_desc', conn, index=False, if_exists='replace')


def upload_prob(
    conn: Connection,
    model_type: str,
    gics_type: str,
    chunk_length: int,
    num_seg: int,
    C: float,
    seed: int = 42
    ) -> None:
    """popluate gics probs for database

    Args:
        conn (Connection): db connection
        model_type (str): language model name
        gics_type (str): industry, industry_group, or sector
        chunk_length (int): max token for a single chunk
        num_seg (int): max token length multiples
        C (float): Inverse of regularization strength of LogisticRegression
        seed (int, optional): seed for LogisticRegression. Defaults to 42.
    """
    tbl_name = f'{model_type}_{gics_type}_{chunk_length * num_seg}_prob'
    conn.execute(create_prob_table_def(tbl_name))

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=C,
        penalty='l2',
        random_state=seed
    )

    texts, labels, tickers = make_input_data(gics_type)
    if model_type == 'bert':
        embeddings = bert_embedding(
            model_name=get_meta_data()['BERT_MODEL_DIR'],
            texts=texts,
            chunk_length=chunk_length,
            num_seg=3
        )

    model.fit(embeddings, labels)
    prob_array = model.predict_proba(embeddings)

    gics_pred = top_classes(
        prob_array=prob_array,
        tickers=tickers,
        gics_type=gics_type,
        top_n=5
    )

    gics_pred.rename({gics_type: 'classification'}, axis=1)
    gics_pred.to_sql(tbl_name, conn, index=False, if_exists='replace')


if __name__ == '__main__':
    db_dir = os.path.join(files('src'), 'data/db')
    conn = sqlite3.connect(os.path.join(db_dir, 'dash.db'))

    upload_prob(
        conn,
        model_type='bert',
        gics_type='industry',
        chunk_length=512,
        num_seg=3,
        C=0.05
    )

    upload_prob(
        conn,
        model_type='bert',
        gics_type='sector',
        chunk_length=512,
        num_seg=3,
        C=0.05
    )


    tbl_def = create_embedding_table_def('bert_embedding_1536', length=768)
    conn.execute(tbl_def)
    conn.execute(CREATE_BUSINESS_DESC_DEF)

    upload_business_desc(conn)
    upload_bert_embedding(conn, num_seg=3)
