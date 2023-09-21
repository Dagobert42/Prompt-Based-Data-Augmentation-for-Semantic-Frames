import argparse
import torch
from helpers.text_processing import *
from helpers.setup import *
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--out_path", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_path", required=True, help="Path to the data.pt files")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=300)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training and validation", default=16)
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

    if args.chunked_training_n > 0:
        with open(args.data_path+"train_data.pt", 'rb') as f:
            X_train, y_train = torch.load(f)
    
    with open(args.data_path+"label_list.pt", 'rb') as f:
        label_list = torch.load(f)
    label_dict = {l: i for i, l in enumerate(label_list)}

    data_collator = DataCollatorForTokenClassification(tokenizer)
    train_data, test_data, val_data = unpack_splits(
        args.data_path,
        label_dict,
        tokenizer,
        args.max_input_length
    )

    model = get_model(args.model, label_list, label_dict)

    model_name = args.out_path.split('/')[-1]
    log_dir = time.strftime("%Y%m%d-%H%M%S") + '_' + model_name
    os.mkdir('./logs/'+ log_dir)

    if args.chunked_training_n > 0:
        train_log = {
                'eval_accuracy' : [],
                'eval_precision' : [],
                'eval_recall' : [],
                'eval_f1' : [],
                'eval_weighted_f1' : [],
                'eval_roc_auc' : [],
                'index' : [],
            }
        trainer, train_log = run_chunked_training(
          X_train,
          y_train,
          test_data,
          train_log,
          label_dict,
          tokenizer,
          args.max_input_length,
          args.chunked_training_n
          )
    else:
        trainer = get_trainer(
            model,
            args.out_path,
            train_data,
            val_data,
            data_collator,
            tokenizer
            )
        trainer.train()
    trainer.save_model(args.out_path+model_name)

    # TODO: use HF save_metrics() ?
    with open(log_dir+"/train_log.pt", 'wb') as f:
        torch.save(train_log, f)

if __name__ == "__main__":
    main()
