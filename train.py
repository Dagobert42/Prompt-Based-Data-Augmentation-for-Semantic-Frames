import argparse
import torch
from helpers.training import *
from helpers.setup import *
from helpers.text_processing import *
from transformers import AutoTokenizer
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--is_local", type=bool, help="Whether to load the model from file or Huggingface", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_base", type=bool, help="Whether to train only the final couple of layers", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_to", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_file", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=512)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training and validation", default=16)
    parser.add_argument("--epochs", type=int, required=False, help="Max number of epochs to train for", default=50)
    parser.add_argument("--chunked_training_n", type=int, required=False, help="Run training in chunks of size n", default=0)
    parser.add_argument("--patience", type=int, required=False, help="Max number of epochs to keep training after eval loss has stopped decreasing", default=2)
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    parser.add_argument("--push_to_hub", type=bool, help="Whether to push the model to Huggingface", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print("CUDA:", torch.cuda.is_available(), torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        add_prefix_space=True
        )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    with open(args.data_file, 'rb') as f:
        splits = torch.load(f)
    train_data, val_data, test_data, label_dict = unpack_splits(
        splits,
        tokenizer,
        args.max_input_length
    )

    model = get_token_classifier(args.model, label_dict, args.is_local, args.freeze_base)
    print(f"Training {args.model} ({model.num_parameters()} params) on {'/'.join(args.data_file.split('/')[-2:])} ({len(train_data)} samples)...")
    n_gpus = len(args.gpu_ids.split(','))

    base_model = args.save_to.split('/')[-3]
    dataset_name = args.save_to.split('/')[-2]
    full_model_name = f"{base_model}-{dataset_name}"
    train_log = {
        'eval_accuracy' : [],
        'eval_precision' : [],
        'eval_recall' : [],
        'eval_f1' : [],
        'eval_weighted_f1' : [],
        'eval_roc_auc' : [],
        'index' : []
        }
    if args.chunked_training_n > 0:
        trainer, train_log = run_chunked_training(
            splits['train']['sentences'],
            splits['train']['labels'],
            test_data,
            model,
            args.save_to,
            val_data,
            train_log,
            label_dict,
            tokenizer,
            args.batch_size,
            n_gpus,
            args.epochs,
            args.patience,
            args.max_input_length,
            args.chunked_training_n
            )
    else:
        trainer = get_trainer(
            model,
            args.save_to+full_model_name,
            train_data,
            val_data,
            tokenizer,
            args.batch_size,
            n_gpus,
            args.epochs,
            args.patience
            )
        trainer.train()
        record_evaluation(trainer, test_data, train_log, len(train_data))
    trainer.save_model(args.save_to)
        
    if args.push_to_hub:
        model_card = {
            "language": "en",
            "license": "mit",
            "tags": ["low-resource NER", "token_classification", "biomedicine", "medical NER"],
            "model_name": f"Dagobert42/{full_model_name}",
            "finetuned_from": base_model,
            "tasks": "ner",
            "dataset_tags": "medicine",
            "dataset": "bigbio/biored"
        }

        trainer.push_to_hub(
            f"Push {args.model} trained on {'-'.join(args.data_file.split('/')[-2:])} ({len(train_data)} samples)",
            **model_card
            )

    # TODO: use HF save_metrics() ?
    log_dir = '../logs/' + time.strftime("%Y%m%d-%H%M%S")
    with open(log_dir+"_train_log.pt", 'wb') as f:
        torch.save(train_log, f)

if __name__ == "__main__":
    main()
