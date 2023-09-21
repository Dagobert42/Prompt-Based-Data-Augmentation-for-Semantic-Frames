import argparse
import os
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from helpers.setup import *
from helpers.text_processing import *
from data.datasets import SequenceClassificationDataset
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np

acc_metric = evaluate.load("accuracy")
pre_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_score = evaluate.load("roc_auc", "multiclass")

def evaluate_sequence_classfication(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)
    results_pre = pre_metric.compute(predictions=preds, references=labels)
    results_rec = rec_metric.compute(predictions=preds, references=labels)
    results_f1 = f1_metric.compute(predictions=preds, references=labels)
    results_acc = acc_metric.compute(predictions=preds, references=labels)
    return {
        'accuracy' : np.round(results_acc['accuracy'], 4),
        'precision' : np.round(results_pre['precision'], 4),
        'recall' : np.round(results_rec['recall'], 4),
        'f1' : np.round(results_f1['f1'], 4),
    }

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--is_local", type=bool, help="Whether to load the model from file or Huggingface", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_to", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_file", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--counter_examples", required=True, help="Path to the .pt file containing the counter examples")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=300)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training and validation", default=8)
    parser.add_argument("--epochs", type=int, required=False, help="Max number of epochs to train for", default=50)
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    assert args.save_to != args.data_file, "Input and output files must be named differently"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.max_input_length,
        add_prefix_space=True
        )
    
    with open(args.data_file, 'rb') as f:
        splits = torch.load(f)
    positive_train = [tag_exemplar(s, l) for s, l in zip(splits['train']['sentences'], splits['train']['labels'])]
    positive_train += [tag_exemplar(s, l) for s, l in zip(splits['val']['sentences'], splits['val']['labels'])]
    positive_test = [tag_exemplar(s, l) for s, l in zip(splits['test']['sentences'], splits['test']['labels'])]

    with open(args.counter_examples, 'rb') as f:
        counter_examples = torch.load(f)
    print
    negative_train, negative_test = train_test_split(
        counter_examples,
        test_size=0.5,
        random_state=1
        )
    X_train = negative_train + positive_train
    y_train = [0] * len(negative_train) + ([1] * len(positive_train))

    X_test = negative_test + positive_test
    y_test = [0] * len(negative_test) + ([1] * len(positive_test))

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)
    train_data = SequenceClassificationDataset(train_encodings, y_train)
    test_data = SequenceClassificationDataset(test_encodings, y_test)

    model = get_sequence_classifier(args.model, args.is_local)
    n_gpus = len(args.gpu_ids.split(','))
    epoch_steps = int(
        len(train_data) / (n_gpus*args.batch_size)
        + (len(train_data) % (n_gpus*args.batch_size) > 0)
        )
    training_args = TrainingArguments(
        args.save_to,
        num_train_epochs=args.epochs,
        evaluation_strategy='steps',
        save_steps=int(epoch_steps/2),
        eval_steps=int(epoch_steps/2),
        save_total_limit=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        push_to_hub=False,
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=evaluate_sequence_classfication,
        callbacks=[EarlyStoppingCallback(3, 0.0)]
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_data)
    trainer.save_model(args.save_to)

if __name__ == "__main__":
    main()
