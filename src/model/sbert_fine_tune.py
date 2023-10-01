"""
Reference:
https://huggingface.co/blog/how-to-train-sentence-transformers
https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss
"""
from random import sample
from typing import Any, Dict, List

from sentence_transformers import (InputExample, SentencesDataset,
                                   SentenceTransformer, losses)
from torch.utils.data import DataLoader

from src.meta_data import get_meta_data
from src.model.model_data import make_input_data


def prepare_data() -> List[InputExample]:
    """prepare training examples for sbert according to 3.3.2

    Returns:
        List[InputExample]: training examples
    """
    texts, labels = make_input_data()
    paired_data = list(zip(texts, labels))
    output = []
    for i in range(len(texts)):
        cls = labels[i]
        text = texts[i]
        # sample 1 example from same class
        tmp_data = [t for t, l in paired_data if l == cls and t != text]
        output.append(
            InputExample(texts=[text, sample(tmp_data, 1)[0]], label=1.0)
        )
        # sample 1 example from different classes
        tmp_data = [t for t, l in paired_data if l != cls]
        output.append(
            InputExample(texts=[text, sample(tmp_data, 1)[0]], label=0.0)
        )

    return output


def tune_sbert(
    num_epochs: int = 3,
    warmup: int = 100,
    batch_size: int = 4,
    opt_params: Dict[str, Any] = {'lr': 2e-5},
    save_model: bool = True,
    **kwargs
) -> None:
    """tune sbert model

    Args:
        num_epochs (int, optional): number of epochs. Defaults to 3.
        warmup (int, optional): number of warmup steps. Defaults to 100.
        batch_size (int, optional): batch size. Defaults to 4.
        opt_params (_type_, optional): optimizer parameters. Defaults to {'lr': 2e-5}.
        save_model (bool, optional): if save model. Defaults to True.
    """
    train_examples = prepare_data()
    model = SentenceTransformer('all-mpnet-base-v2')
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    train_loss = losses.CosineSimilarityLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup,
        optimizer_params=opt_params,
        **kwargs
    )
    if save_model:
        model_dir = get_meta_data()['SBERT_MODEL_DIR']
        model.save(path=model_dir)


if __name__ == '__main__':
    tune_sbert(save=True)