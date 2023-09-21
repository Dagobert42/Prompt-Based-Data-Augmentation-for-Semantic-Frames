import torch
import os
from data.datasets import TokenClassificationDataset, SequenceClassificationDataset
from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification

def unpack_splits(
        splits,
        tokenizer,
        max_input_length
        ):
    label_dict = {l: i for i, l in enumerate(splits["label_list"])}

    train_data = TokenClassificationDataset(
        splits["train"]["sentences"],
        splits["train"]["labels"],
        label_dict,
        tokenizer,
        max_input_length
        )
    val_data = TokenClassificationDataset(
        splits["val"]["sentences"],
        splits["val"]["labels"],
        label_dict,
        tokenizer,
        max_input_length
        )
    test_data = TokenClassificationDataset(
        splits["test"]["sentences"],
        splits["test"]["labels"],
        label_dict,
        tokenizer,
        max_input_length
        )
    
    return train_data, val_data, test_data, label_dict


def get_token_classifier(model_path, label_dict, is_local, freeze_base=False):
    if is_local:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(label_dict.keys()),
            id2label={str(i): key for key, i in label_dict.items()},
            label2id=label_dict
        )
        model = AutoModelForTokenClassification.from_config(config)
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False
    return model


def get_sequence_classifier(model_path, is_local):
    if is_local:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_config(config)
    return model
