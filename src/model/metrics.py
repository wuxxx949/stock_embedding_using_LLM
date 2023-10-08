from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TextClassificationPipeline


def classification_metrics(
    preds: np.array,
    labels: np.array
) -> Dict[str, float]:
    """evaluate performance for multinomial logistic regression

    Args:
        preds (List[int]): predicted labels
        labels (List[int]): actual labels

    Returns:
        Dict[str, float]: _description_
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
