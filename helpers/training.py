import evaluation
from tqdm import tqdm
import numpy as np
from os import system
from data.datasets import TokenClassificationDataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback

def get_trainer(
        model,
        save_to,
        train_data,
        val_data,
        tokenizer,
        batch_size,
        n_gpus,
        epochs,
        patience
        ):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    epoch_steps = int(len(train_data) / (n_gpus*batch_size) + (len(train_data) % (n_gpus*batch_size) > 0))
    args = TrainingArguments(
        save_to,
        evaluation_strategy='steps',
        eval_steps=epoch_steps,
        save_steps=epoch_steps,
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True
    )
    return Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=evaluation.compute_metrics,
        callbacks=[EarlyStoppingCallback(patience, 0.0)]
    )

def record_evaluation(trainer, test_data, train_log, at_samples):
    evaluation = trainer.evaluate(eval_dataset=test_data)
    for key, value in evaluation.items():
        if key in train_log.keys():
            train_log[key].append(value)
    train_log['index'].append(at_samples)
    print(f"Evaluation after training {trainer.model.config_class.model_type} on {at_samples} samples: \n{evaluation}")

def run_chunked_training(
        X,
        y,
        test_data,
        model,
        save_to,
        val_data,
        train_log,
        label_dict,
        tokenizer,
        batch_size,
        n_gpus,
        epochs,
        patience,
        max_input_length,
        n_per_training
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
        trainer = get_trainer(
            model,
            save_to,
            data_chunk,
            val_data,
            tokenizer,
            batch_size,
            n_gpus,
            epochs,
            patience
            )
        record_evaluation(trainer, test_data, train_log, from_idx)
        trainer.train()
        system('clear')

    record_evaluation(trainer, test_data, train_log, n_samples)
    return trainer, train_log