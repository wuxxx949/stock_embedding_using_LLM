from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.model.metrics import classification_metrics


def model_eval(
    X: np.array,
    y: np.array,
    C: float = 1.0,
    test_size: float = 0.2,
    seed: int = 42
    ) ->  Dict[str, float]:
    """evaluate embeddings by fitting a multinomial logistic regression

    Args:
        X (np.array): embeddings as feature
        y (np.array): labels
        C (float, optional): Inverse of regularization strength. Defaults to 1.0.
        test_size (float, optional): test size. Defaults to 0.2.
        seed (int, optional): for random state. Defaults to 42.

    Returns:
        Dict[str, float]: _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=C,
        penalty='l2',
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    return classification_metrics(y_pred, y_test)


if __name__ == '__main__':
    from src.meta_data import get_meta_data
    from src.model.embedding import bert_embedding, sbert_embedding
    from src.model.model_data import make_input_data


    texts, labels, _ = make_input_data()
    embeddings = bert_embedding(
        model_name=get_meta_data()['BERT_MODEL_DIR'],
        texts=texts,
        chunk_length=510,
        num_seg=3
    )

    model_result = model_eval(X=embeddings, y=labels)

    sbert_embeddings = sbert_embedding(
        model_name=get_meta_data()['SBERT_MODEL_DIR'],
        texts=texts,
        chunk_length=382,
        num_seg=3
    )

    model_result = model_eval(X=sbert_embeddings, y=labels, C=6.0)