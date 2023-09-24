import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from src.model.model_data import make_input_data
from src.meta_data import get_meta_data

meta_data = get_meta_data()
freeze_encode = True

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


if __name__ == '__main__':
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    texts, labels = make_input_data()
    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(texts, labels, test_size=.2, stratify=labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)


    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(set(labels)))

    # freeze encoder first
    if freeze_encode:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        learning_rate=0.01,
        num_train_epochs=15,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.evaluate()

    # unfreeze encode and train again
    for param in model.base_model.parameters():
        param.requires_grad = True


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        learning_rate=2e-5,
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )
    trainer.train()
    trainer.evaluate()

    model.save_pretrained(meta_data['MODEL_DIR'])
