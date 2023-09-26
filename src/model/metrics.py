import re

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TextClassificationPipeline


def compute_metrics(preds, labels):
    precision, recall, f1, _ = \
          precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def truncate(sentences, tokenizer, max_length):
    # Tokenize and truncate all sentences using list comprehension
    truncated_sentences = [
        tokenizer.convert_tokens_to_string(tokenizer.tokenize(sentence)[:max_length])
        for sentence in sentences
    ]

    return truncated_sentences

def make_pred(model, tokenizer, sentences, max_length):
    texts = truncate(sentences, tokenizer, max_length)

    pipe = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, return_all_scores=False
    )
    pred = pipe(texts)
    pred_labels = [int(re.search('\d+', d['label']).group()) for d in pred]

    return pred_labels



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.meta_data import get_meta_data
    from src.model.model_data import make_input_data

    texts, labels = make_input_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        get_meta_data()['MODEL_DIR']
    )
    _, val_texts, _, val_labels = \
        train_test_split(texts, labels, test_size=0.2, random_state=21, stratify=labels)
    checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    pred_labels = make_pred(model, tokenizer, val_texts, max_length=510)
    out = compute_metrics(preds=pred_labels, labels=val_labels)