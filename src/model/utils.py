from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.models.mpnet.tokenization_mpnet_fast import \
    MPNetTokenizerFast

from src.model.model_data import make_input_data


def text_segment(
    tokenizer: MPNetTokenizerFast | BertTokenizerFast,
    text: str,
    text_id: int,
    chunk_length: int,
    num_seg: int
) -> Tuple[List[str], List[int]]:
    """break text into segments if longer than model token limit

    Args:
        tokenizer (MPNetTokenizerFast|BertTokenizerFast): tokenizer object
        text (str): text input
        text_id (int): text id
        chunk_length (int): size of each segment
        num_seg (int): number of segments

    Returns:
        Tuple[List[str], List[int]]: _description_
    """
    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Break the tokens into chunks of size max_chunk_length
    text_output = []
    id_output = []
    text_len = num_seg * chunk_length
    for i in range(0, text_len, chunk_length):
        tokens_chunk = tokens[i:i + chunk_length]
        if len(tokens_chunk) == chunk_length:
            text_output.append(tokenizer.decode(tokens_chunk))
            id_output.append(text_id)
        else:
            break

    return text_output, id_output


def mean_embedding(embeddings: np.array, text_id: np.array) -> np.array:
    """calcuate mean embeddings

    Args:
        embeddings (np.array): embeddings with text_size x dim
        text_id (np.array): text id as grouping index

    Returns:
        np.array: mean embeddings
    """
    unique_ids = np.unique(text_id)
    mean_vec = [np.mean(embeddings[text_id == group], axis=0) for group in unique_ids]

    return np.vstack(mean_vec)


