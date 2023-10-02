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
    parser.add_argument("--out_path", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--augmentations_path", required=True, help="Path to the .pt file containing the data splits")
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    assert args.out_path != args.augmentations_path, "Input and output files must be named differently"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    with open(args.augmentations_path, 'rb') as f:
        splits = torch.load(f)
    augmentations = [tag_exemplar(s, l) for s, l in zip(splits['train']['sentences'], splits['train']['labels'])]

    classifier = pipeline(
        "sentiment-analysis",
        model=args.model,
        padding=True,
        truncation=True
        )
    scores = classifier(augmentations)

    filtered_augmentations = [
        parse_markup(a, splits['label_list'])
        for (s, a) in zip(scores, augmentations)
        if s['label']=='LABEL_1'
        and s['score']>=0.5
        ]
    
    splits['train']['sentences'] = [a[0] for a in filtered_augmentations]
    splits['train']['labels'] = [a[1] for a in filtered_augmentations]

    with open(args.out_path, 'wb') as f:
        torch.save(splits, f)
