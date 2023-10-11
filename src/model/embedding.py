"""make embedings for fine-tuned models
"""
import multiprocessing as mp
from functools import reduce
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.meta_data import get_meta_data
from src.model.utils import mean_embedding, text_segment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        return {"text": text}


def text_preprocessing(
    tokenizer,
    texts: List[str],
    chunk_length: int,
    num_seg: int,
    ncores: int | None = None
) -> Tuple[List[str], List[int]]:
    """_summary_

    Args:
        tokenizer: tokenizer of the assoicated model
        texts (List[str]): texts to embed
        chunk_length (int): text chunk length
        num_seg (int): number of segments to average over
        ncores (Optional[int], optional): threads for mp. Defaults to None.

    Returns:
        Tuple[List[str], List[int]]: _description_
    """
    ncores = ncores if ncores is not None else mp.cpu_count()
    params_lst = [(tokenizer, text, i, chunk_length, num_seg) for i, text in enumerate(texts)]
    with mp.Pool(processes=ncores) as p:
        out = p.starmap(text_segment, params_lst)

    seg_texts = []
    seg_ids = []
    for e1, e2 in out:
        seg_texts.extend(e1)
        seg_ids.extend(e2)

    return seg_texts, seg_ids


def bert_embedding(
    model_name: str,
    texts: List[str],
    chunk_length: int,
    num_seg: int,
    ncores: int | None = None,
    embedding_type: str = 'cls'
) -> np.array:
    """bert embedding inference

    Args:
        model_name (str): model name or local model path
        texts (List[str]): texts to embed
        chunk_length (int): text chunk length
        num_seg (int): number of segments to average over
        ncores (Optional[int], optional): threads for mp. Defaults to None.
        embedding_type (str): cls or pooler

    Returns:
        np.array: embeddings as np.array with shape text_count x dim
    """

    ft_model = AutoModelForSequenceClassification.from_pretrained(model_name).bert
    ft_model.to(device)

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    seg_texts, seg_ids = text_preprocessing(
        tokenizer=tokenizer,
        texts=texts,
        chunk_length=chunk_length,
        num_seg=num_seg,
        ncores=ncores
    )

    dataset = CustomDataset(list(seg_texts))
    # Use DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    embeddings_lst = []
    for batch in dataloader:
        # Extract texts and labels from the batch
        texts_batch = batch["text"]

        # Tokenize and pad the batch of sentences
        inputs = tokenizer(
            texts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs.to(device)

        with torch.no_grad():
            outputs = ft_model(**inputs)
        last_hidden_state, pooler_output = outputs[0], outputs[1]
        if embedding_type == 'cls':
            embedding = last_hidden_state[:, 0, :]
        else:
            embedding = pooler_output
        if device.type == 'cuda':
            embeddings_lst.extend(embedding.cpu().numpy())

    raw_embeddings = np.vstack(embeddings_lst)
    embeddings = mean_embedding(raw_embeddings, seg_ids)

    return embeddings


def sbert_embedding(
    model_name: str,
    texts: List[str],
    chunk_length: int,
    num_seg: int,
    ncores: int | None = None
) -> np.array:
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    seg_texts, seg_ids = text_preprocessing(
        tokenizer=tokenizer,
        texts=texts,
        chunk_length=chunk_length,
        num_seg=num_seg,
        ncores=ncores
    )

    # cpu for inference
    raw_embeddings = model.encode(seg_texts)
    embeddings = mean_embedding(raw_embeddings, seg_ids)

    return embeddings


if __name__ == '__main__':
    from src.model.model_data import make_input_data

    texts, labels, _ = make_input_data()
    model_path = get_meta_data()['BERT_MODEL_DIR']
    test_embeddings = bert_embedding(
        model_name=model_path,
        texts=texts,
        chunk_length=510,
        num_seg=3
    )

