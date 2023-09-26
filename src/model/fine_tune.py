import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.models.bert.modeling_bert import \
    BertForSequenceClassification

from src.meta_data import get_meta_data
from src.model.model_data import make_input_data


# https://huggingface.co/transformers/v4.2.2/training.html
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key != 'token_type_ids'
            }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def fine_tune_bert(
    test_size: float = 0.2,
    seed: int = 21,
    lr_freeze: float = 0.01,
    lr_unfreeze: float = 2e-5,
    num_epochs_freeze  = 15,
    num_epochs_unfreeze = 5,
    save_model: bool = True
) -> BertForSequenceClassification:
    """fine tune a bert model:
        1. freeze the base model and train softmax layer with large rate
        2. unfreeze the base model and train the whole model with small rate

    Args:
        test_size (float): test set size between 0 and 1
        seed (int): random seed for training test split
        lr_freeze (float, optional): learning rate for step 1. Defaults to 0.01.
        lr_unfreeze (float, optional): learning rate for step 2. Defaults to 2e-5.
        num_epochs_freeze (int, optional): num of epochs for step 1. Defaults to 15.
        num_epochs_unfreeze (int, optional): num of epochs for step 2. Defaults to 5.
        save_model (bool, optional): if save model to folder. Defaults to True.

    Returns:
        BertForSequenceClassification: tuned model
    """
    checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    texts, labels = make_input_data()

    if 0 < test_size < 1:
        train_texts, val_texts, train_labels, val_labels = \
            train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        val_dataset = CustomDataset(val_encodings, val_labels)
    elif test_size == 0:
        train_texts, train_labels = texts, labels
        val_dataset = None
    else:
        raise ValueError('test size outside of [0, 1)')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(set(labels))
    )

    # freeze encoder first
    for param in model.base_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=lr_freeze,
        num_train_epochs=num_epochs_freeze,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    # unfreeze encode and train again
    for param in model.base_model.parameters():
        param.requires_grad = True


    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=lr_unfreeze,
        num_train_epochs=num_epochs_unfreeze,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    if save_model:
        meta_data = get_meta_data()
        model.save_pretrained(meta_data['MODEL_DIR'])

    return model


if __name__ == '__main__':
    tuned_model = fine_tune_bert(save_model=True)

    checkpoint = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(
        get_meta_data()['MODEL_DIR']
    )
