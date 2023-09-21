
import json
from sklearn.model_selection import train_test_split
import time
import torch

def load_data():
    sentences = []
    all_labels = []
    with open("ner_annotations.json", "rb") as f:
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
    # maybe TODO: expose split values, data path, random state via args

    sentences, all_labels, label_list = load_data()

    X_train, X_rest, y_train, y_rest = train_test_split(
        sentences,
        all_labels,
        test_size=0.66666,
        random_state=1
        )

    X_test, X_val, y_test, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        random_state=1
        )
    
    with open("./train_data.pt", 'wb') as f:
        torch.save((X_train, y_train), f)
    with open("./test_data.pt", 'wb') as f:
        torch.save((X_test, y_test), f)
    with open("./val_data.pt", 'wb') as f:
        torch.save((X_val, y_val), f)
    with open("./label_list.pt", 'wb') as f:
        torch.save(label_list, f)
    

if __name__ == "__main__":
    main()
