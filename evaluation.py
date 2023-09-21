import numpy as np
import evaluate
import warnings
from helpers.setup import *
from sklearn import metrics
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from data.datasets import TokenClassificationDataset
import matplotlib.pyplot as plt
from scipy.special import softmax
import torch
import argparse
from sklearn.metrics import ConfusionMatrixDisplay
import os
import time

acc_metric = evaluate.load("accuracy")
pre_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_score = evaluate.load("roc_auc", "multiclass")


def compute_base_metrics(flat_preds, flat_labels):
    results_acc = acc_metric.compute(predictions=flat_preds, references=flat_labels)
    results_pre = pre_metric.compute(predictions=flat_preds, references=flat_labels, average="macro", zero_division=0)
    results_rec = rec_metric.compute(predictions=flat_preds, references=flat_labels, average="macro", zero_division=0)
    results_f1 = f1_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")
    results_wf1 = f1_metric.compute(predictions=flat_preds, references=flat_labels, average="weighted")
    return {
        'accuracy' : np.round(results_acc['accuracy'], 4),
        'precision' : np.round(results_pre['precision'], 4),
        'recall' : np.round(results_rec['recall'], 4),
        'f1' : np.round(results_f1['f1'], 4),
        'weighted_f1' : np.round(results_wf1["f1"], 4),
    }


def compute_roc_aucs(logits, flat_id_labels):
    # logits are necessary for roc auc
    flat_logits = softmax([k for o in logits for k in o], axis=1)
    return roc_auc_score.compute(
        references=flat_id_labels,
        prediction_scores=flat_logits,
        multi_class='ovr',
        average=None
        )


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=2)
    
    # Remove ignored index (special tokens)
    id_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    id_labels = [
        [l for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_preds = [i for m in id_predictions for i in m]
    flat_labels = [j for n in id_labels for j in n]
    return compute_base_metrics(flat_preds, flat_labels)


def get_predictions(trainer, samples, label_list):
    all_logits, labels, _ = trainer.predict(samples)
    predictions = np.argmax(all_logits, axis=2)

    # Remove ignored index (special tokens)
    logits = [
        [token_lgts for (token_lgts, l) in zip(instance_lgts, label) if l != -100]
        for instance_lgts, label in zip(all_logits, labels)
    ]
    id_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    id_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Replace ids with actual classes
    real_predictions = [
        [label_list[p] for p in prediction]
        for prediction in id_predictions
    ]
    real_labels = [
        [label_list[l] for l in label]
        for label in id_labels
    ]
    return logits, id_predictions, id_labels, real_predictions, real_labels


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--out_path", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_path", required=True, help="Path to the data.pt files")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=300)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for evaluation", default=16)
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.max_input_length,
        add_prefix_space=True
        )
    
    with open(args.data_path+"label_list.pt", 'rb') as f:
        label_list = torch.load(f)
    label_dict = {l: i for i, l in enumerate(label_list)}

    with open(args.data_path+"test_data.pt", 'rb') as f:
        X_test, y_test = torch.load(f)
    test_data = TokenClassificationDataset(
        X_test,
        y_test,
        label_dict,
        tokenizer,
        args.max_input_length
        )
    
    model = get_model(args.model, label_list, label_dict)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(per_device_eval_batch_size=args.batch_size)
    trainer = Trainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=evaluation.compute_metrics
    )

    eval_log = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (
            logits,
            id_predictions,
            id_labels,
            real_predictions,
            real_labels
        ) = get_predictions(trainer, test_data, label_list)
    
    flat_id_preds = [i for m in id_predictions for i in m]
    flat_id_labels = [j for n in id_labels for j in n]
    base_metrics = compute_base_metrics(flat_id_preds, flat_id_labels)
    for key, value in base_metrics.items():
        if key in eval_log.keys():
            eval_log[key] = value
    print(base_metrics)

    classwise_roc_aucs = compute_roc_aucs(logits, flat_id_labels)
    eval_log['roc_aucs'] = classwise_roc_aucs
    print(classwise_roc_aucs)

    flat_real_predictions = [i for m in real_predictions for i in m]
    flat_real_labels = [j for n in real_labels for j in n]
    cm_recall = metrics.confusion_matrix(flat_real_labels, flat_real_predictions, labels=label_list, normalize="true")
    cm_precision = metrics.confusion_matrix(flat_real_labels, flat_real_predictions, labels=label_list, normalize="pred")
    eval_log['cm_recall'] = cm_recall
    eval_log['cm_precision'] = cm_precision

    model_name = args.out_path.split('/')[-1]
    log_dir = time.strftime("%Y%m%d-%H%M%S") + '_' + model_name
    os.mkdir('./logs/'+ log_dir)
    with open(log_dir+"/eval_log.pt", 'wb') as f:
        torch.save(eval_log, f)

    cm_sum = cm_precision + cm_recall
    cm_sum[cm_sum==0] = 1 # just to avoid division by 0
    cm_f1 = (cm_precision*cm_recall)/cm_sum
    cm = np.round(cm_f1, 2)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_list).plot(xticks_rotation='vertical')
    fig = disp.figure_

    disp.ax_.set_title(model_name, {'fontsize': 20})
    fig.set_figwidth(18)
    fig.set_figheight(18)
    plt.show()
    fig.savefig(f"./images/{model_name}-results.png", dpi=300, format="png")

if __name__ == "__main__":
    main()
