import argparse
import json
from sklearn.model_selection import train_test_split
import torch

def load_data(path):
    sentences = []
    all_labels = []
    with open(path, "rb") as f:
        for paper in json.load(f)["data"]:
            sentences += paper["tokens"]
            all_labels += paper["labels"]

    label_list = []
    for labels in all_labels:
        for label in labels: # sentence level
            if label not in label_list:
                label_list.append(label)

    return sentences, all_labels, label_list


def main():
    parser = argparse.ArgumentParser(description="Create material synthesis data splits")

    parser.add_argument("--save_to", required=True, help="Name or path of the .pt file to save the dataset splits to")
    parser.add_argument("--test_split", type=float, required=False, help="Overall portion of the test split [0-1]", default=0.66666)
    args = parser.parse_args()
    sentences, all_labels, label_list = load_data("ner_annotations.json")

    X_train, X_rest, y_train, y_rest = train_test_split(
        sentences,
        all_labels,
        test_size=args.test_split,
        random_state=1
        )

    X_test, X_val, y_test, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        random_state=1
        )
    
    splits = {
        'train' : { 'sentences' : X_train, 'labels' : y_train },
        'val' : { 'sentences' : X_val, 'labels' : y_val },
        'test' : { 'sentences' : X_test, 'labels' : y_test },
        'label_list' : label_list
    }
    with open(args.save_to, 'wb') as f:
        torch.save(splits, f)


if __name__ == "__main__":
    main()
