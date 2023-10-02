import evaluation
from tqdm import tqdm
import numpy as np
# from IPython.display import clear_output
from os import system
from data.datasets import TokenClassificationDataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback

def get_trainer(
        model,
        out_path,
        train_data,
        val_data,
        tokenizer,
        batch_size,
        epochs
        ):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    epoch_steps = int(len(train_data) / batch_size + (len(train_data) % batch_size > 0))
    args = TrainingArguments(
        out_path,
        evaluation_strategy='steps',
        eval_steps=epoch_steps,
        save_steps=epoch_steps,
        # eval_delay=3*epoch_steps,
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
        callbacks=[EarlyStoppingCallback(1, 0.0)]
    )

def record_evaluation(trainer, test_data, train_log, at_samples):
    evaluation = trainer.evaluate(eval_dataset=test_data)
    for key, value in evaluation.items():
        if key in train_log.keys():
            train_log[key].append(value)
    train_log['index'].append(at_samples)
    print(f"Evaluation after training on {at_samples} samples: \n{evaluation}")

def run_chunked_training(
        X,
        y,
        test_data,
        model,
        out_path,
        val_data,
        train_log,
        label_dict,
        tokenizer,
        batch_size,
        epochs,
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
            out_path,
            data_chunk,
            val_data,
            tokenizer,
            batch_size,
            epochs
            )
        record_evaluation(trainer, test_data, train_log, from_idx)
        trainer.train()
        system('clear')

    record_evaluation(trainer, test_data, train_log, n_samples)
    return trainer, train_log