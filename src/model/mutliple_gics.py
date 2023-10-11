# assign multiple gics
from typing import List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model.model_data import (gics_code_to_string, make_gics_look_up,
                                  make_input_data)


def top_classes(
    prob_array: np.array,
    tickers: List[str],
    gics_type: str,
    top_n: int = 3
) -> pd.DataFrame:
    gics_code_mapping = gics_code_to_string(gics_type)
    gics_lookup, int_to_gics_mapping = make_gics_look_up(gics_type)
    # Get the indices of the top 3 classes for each example
    top_classes_indices = np.argsort(prob_array, axis=1)[:, ::-1][:, :top_n]
    top_classes_prob = np.take_along_axis(prob_array, top_classes_indices, axis=1)
    gics_lookup

    out = []
    for  indices, probs, ticker in (zip(top_classes_indices, top_classes_prob, tickers)):
        for i, prob in zip(indices, probs):
            tmp_gics_code =  int_to_gics_mapping[i]
            tmp_gics = gics_code_mapping[tmp_gics_code]
            out.append((ticker, tmp_gics, prob))

    prob_df = pd.DataFrame(out, columns=['ticker', gics_type, 'prob'])

    return prob_df


if __name__ == '__main__':
    from src.meta_data import get_meta_data
    from src.model.embedding import bert_embedding, sbert_embedding
    from src.model.model_data import make_input_data

    gics_type = 'industry'
    texts, labels, tickers = make_input_data(gics_type)
    embeddings = bert_embedding(
        model_name=get_meta_data()['BERT_MODEL_DIR'],
        texts=texts,
        chunk_length=382,
        num_seg=3
    )

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=0.01,
        penalty='l2',
        random_state=42
    )

    model.fit(embeddings, labels)

    prob_array = model.predict_proba(embeddings)

    gics_pred = top_classes(
        prob_array=prob_array,
        tickers=tickers,
        gics_type=gics_type,
        top_n=5
    )

    gics_pred.query("ticker=='nflx'")
