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
    parser.add_argument("--out_path", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_path", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=300)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training and validation", default=16)
    parser.add_argument("--epochs", type=int, required=False, help="Max number of epochs to train for", default=50)
    parser.add_argument("--chunked_training_n", type=int, required=False, help="Run training in chunks of size n", default=0)
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.max_input_length,
        add_prefix_space=True
        )

    with open(args.data_path, 'rb') as f:
        splits = torch.load(f)

    train_data, val_data, test_data, label_dict = unpack_splits(
        splits,
        tokenizer,
        args.max_input_length
    )

    model = get_token_classifier(args.model, label_dict, args.is_local, args.freeze_base)

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
            args.out_path,
            val_data,
            train_log,
            label_dict,
            tokenizer,
            args.batch_size,
            args.epochs,
            args.max_input_length,
            args.chunked_training_n
            )
    else:
        trainer = get_trainer(
            model,
            args.out_path,
            train_data,
            val_data,
            tokenizer,
            args.batch_size,
            args.epochs
            )
        trainer.train()
        record_evaluation(trainer, test_data, train_log, len(train_data))
    trainer.save_model(args.out_path)

    # TODO: use HF save_metrics() ?
    log_dir = './logs/' + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(log_dir)
    with open(log_dir+"/train_log.pt", 'wb') as f:
        torch.save(train_log, f)

if __name__ == "__main__":
    main()
