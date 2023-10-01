"""make embedings for fine-tuned models
"""
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.meta_data import get_meta_data
from src.model.model_data import make_input_data


class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {"text": text, "label": label}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

def bert_embedding(
    model_name: str,
    texts: List[str],
    labels: List[int]
) -> Tuple[List[List[float]], List[int]]:
    # model_name = get_meta_data()['BERT_MODEL_DIR']
    ft_model = AutoModelForSequenceClassification.from_pretrained(model_name).bert
    ft_model.to(device)

    dataset = CustomDataset(texts, labels)
    # Use DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained(model_name)

    # Iterate through batches
    embeddings_lst = []
    labels_lst = []
    for batch in dataloader:
        # Extract texts and labels from the batch
        texts_batch = batch["text"]
        labels_batch = batch["label"]

        # Tokenize and pad the batch of sentences
        inputs = tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs.to(device)

        # Perform batched inference (modify this part based on your specific model)
        with torch.no_grad():
            # Assuming model is an instance of your BERT model
            outputs = ft_model(**inputs)
        last_hidden_state, pooler_output = outputs[0], outputs[1]
        cls_embedding = last_hidden_state[:, 0, :]
        if device.type == 'cuda':
            embeddings_lst.extend(cls_embedding.cpu().tolist())
        labels_lst.extend(labels_batch.tolist())

    return embeddings_lst, labels_lst

if __name__ == '__main__':
    texts, labels = make_input_data()
    model_path = get_meta_data()['BERT_MODEL_DIR']
    embeddings, labels = bert_embedding(model_path, texts, labels)
