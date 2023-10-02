import argparse
import os
from transformers import AutoTokenizer, TrainingArguments, Trainer, pipeline
from helpers.setup import *
from helpers.text_processing import *
from sklearn.model_selection import train_test_split
from evaluation import *

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--is_local", type=bool, help="Whether to load the model from file or Huggingface", action=argparse.BooleanOptionalAction)
    parser.add_argument("--out_path", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--data_path", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--counter_examples", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--max_input_length", type=int, required=False, help="Max number of tokens in a sequence", default=300)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training and validation", default=8)
    parser.add_argument("--epochs", type=int, required=False, help="Max number of epochs to train for", default=5)
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    assert args.out_path != args.data_path, "Input and output files must be named differently"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.max_input_length,
        add_prefix_space=True
        )
    
    with open(args.data_path, 'rb') as f:
        splits = torch.load(f)
    positive_train = [tag_exemplar(s, l) for s, l in zip(splits['train']['sentences'], splits['train']['labels'])]
    positive_train += [tag_exemplar(s, l) for s, l in zip(splits['val']['sentences'], splits['val']['labels'])]
    positive_test = [tag_exemplar(s, l) for s, l in zip(splits['test']['sentences'], splits['test']['labels'])]

    with open(args.counter_examples, 'rb') as f:
        counter_examples = torch.load(f)
    sentences, labels = extract_augmentations(counter_examples, splits['label_list'])
    counter_examples = [tag_exemplar(s, l) for s, l in zip(sentences, labels)]

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

    training_args = TrainingArguments(
        args.out_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        compute_metrics=evaluate_sequence_classfication
    )

    trainer.train()
    print(trainer.evaluate(eval_dataset=test_data))
    trainer.save_model(args.out_path)
