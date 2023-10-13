import os
import sqlite3

import pandas as pd

from src.meta_data import get_meta_data
from src.model.embedding import bert_embedding, sbert_embedding
from src.model.model_data import (gics_code_to_string, make_gics_look_up,
                                  make_input_data)
from src.data.db.queries import CREATE_BUSINESS_DESC_QUERY


def create_embedding_table_query(tbl_name: str, length: int) -> str:
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


def upload_bert_embedding(conn: sqlite3.Connection, num_seg: int) -> None:
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



if __name__ == '__main__':
    db_dir = os.path.dirname(os.path.realpath(__file__))
    conn = sqlite3.connect(os.path.join(db_dir, 'dash.db'))

    tbl_def = create_embedding_table_query('bert_embedding_1536', length=768)
    conn.execute(tbl_def)
    conn.execute(CREATE_BUSINESS_DESC_QUERY)

    upload_bert_embedding(conn, num_seg=3)

