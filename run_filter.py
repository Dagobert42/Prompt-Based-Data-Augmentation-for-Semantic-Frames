import argparse
import os
import time
import torch
from transformers import pipeline
from helpers.setup import *
from helpers.text_processing import *

def main():
    parser = argparse.ArgumentParser(description="Baseline Model Training")

    parser.add_argument("--model", required=True, help="Name or path of the initial Hugging Face model to load")
    parser.add_argument("--out_folder", required=True, help="Path for the trained model to save weights and logs")
    parser.add_argument("--augmentations_file", required=True, help="Path to the folder to save the filtered splits to")
    parser.add_argument("--thresholds", type=str, required=False, help="Thresholds for the filtering to create splits at", default="0.5")
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    with open(args.augmentations_file, 'rb') as f:
        splits = torch.load(f)
    augmentations = [tag_exemplar(s, l) for s, l in zip(splits['train']['sentences'], splits['train']['labels'])]

    classifier = pipeline(
        "sentiment-analysis",
        model=args.model,
        padding=True,
        truncation=True
        )
    print(f"Running inference on {args.augmentations_file}...")
    scores = classifier(augmentations)

    for thresh in args.thresholds.split(','):
        filtered_augmentations = [
            parse_markup(a, splits['label_list'])
            for (s, a) in zip(scores, augmentations)
            if s['label']=='LABEL_1'
            or (s['label']=='LABEL_0' and s['score']<float(thresh))
            ]
        
        splits['train']['sentences'] = [a[0] for a in filtered_augmentations]
        splits['train']['labels'] = [a[1] for a in filtered_augmentations]

        with open(args.out_folder+f"filtered_splits-{str(len(filtered_augmentations))}.pt", 'wb') as f:
            torch.save(splits, f)
        print(f"Saved filtered_splits-{str(len(filtered_augmentations))}.pt for threshold value {thresh}...")

if __name__ == "__main__":
    main()
