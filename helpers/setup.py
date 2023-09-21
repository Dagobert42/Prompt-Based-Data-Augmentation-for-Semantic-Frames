import torch
import eval
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
from data.datasets import TokenClassificationDataset
from transformers import TrainingArguments, Trainer
from transformers import AutoConfig, AutoModelForTokenClassification


def unpack_splits(data_path, label_dict, tokenizer, max_input_length):
    
    with open(data_path+"train_data.pt", 'rb') as f:
        X_train, y_train = torch.load(f)
    train_data = TokenClassificationDataset(
        X_train,
        y_train,
        label_dict,
        tokenizer,
        max_input_length
        )
    
    with open(data_path+"test_data.pt", 'rb') as f:
        X_test, y_test = torch.load(f)
    test_data = TokenClassificationDataset(
        X_test,
        y_test,
        label_dict,
        tokenizer,
        max_input_length
        )
    
    with open(data_path+"val_data.pt", 'rb') as f:
        X_val, y_val = torch.load(f)
    val_data = TokenClassificationDataset(
        X_val,
        y_val,
        label_dict,
        tokenizer,
        max_input_length
        )
    
    return train_data, test_data, val_data

def get_model(model_path, label_list, label_dict):
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(label_list),
            id2label={str(i): key for key, i in label_dict.items()},
            label2id={key: i for key, i in label_dict.items()}
        )
        model = AutoModelForTokenClassification.from_config(config)
        for param in model.base_model.parameters():
            param.requires_grad = False
    except:
        model = AutoModelForTokenClassification.from_pretrained(model_path)

def get_trainer(
        model,
        out_path,
        train_data,
        val_data,
        data_collator,
        tokenizer
        ):
    batch_size = 16
    args = TrainingArguments(
        out_path,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # TODO: early stopping
        num_train_epochs=10,
        weight_decay=0.01,
    )

    return Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=eval.compute_metrics
    )

def record_evaluation(trainer, test_data, train_log, at_samples):
    evaluation = trainer.evaluate(eval_dataset=test_data)
    for key, value in evaluation.items():
        if key in train_log.keys():
            train_log[key].append(value)
    train_log['index'].append(at_samples)
    print('Evaluation:', evaluation)

def run_chunked_training(
        X,
        y,
        test_data,
        train_log,
        label_dict,
        tokenizer,
        max_input_length,
        n_per_training=50
        ):
    n_samples = len(X)
    for i in tqdm(range(int(np.ceil(n_samples/n_per_training)))):
        from_idx = i*n_per_training
        to_idx = min((i+1)*n_per_training, n_samples)
        data_chunk = TokenClassificationDataset(
            X[from_idx:to_idx],
            y[from_idx:to_idx],
            label_dict,
            tokenizer,
            max_input_length
            )
        trainer = get_trainer(data_chunk)
        record_evaluation(trainer, test_data, train_log, from_idx)
        trainer.train()
        clear_output()

    record_evaluation(trainer, train_log, n_samples)
    return trainer, train_log