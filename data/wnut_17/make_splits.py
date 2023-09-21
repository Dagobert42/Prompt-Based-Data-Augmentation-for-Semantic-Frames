import argparse
import torch
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Create WNUT 17 data splits")
    parser.add_argument("--train_size", type=int, required=False, help="Target size of train split, potential leftovers will be added to test split", default=200)
    parser.add_argument("--save_to", required=True, help="Name or path of the .pt file to save the dataset splits to")
    args = parser.parse_args()

    dataset = load_dataset("wnut_17")

    # labels were included with the huggingface model card
    # we define them manually and remove the IOB formatting
    # label_dict = {
    #     0 : "null",
    #     1 : "corporation",
    #     2 : "corporation",
    #     3 : "creative-work",
    #     4 : "creative-work",
    #     5 : "group",
    #     6 : "group",
    #     7 : "location",
    #     8 : "location",
    #     9 : "person",
    #     10 : "person",
    #     11 : "product",
    #     12 : "product"
    # }

    id2labels = {
        0: "O",
        1: "B-corporation",
        2: "I-corporation",
        3: "B-creative-work",
        4: "I-creative-work",
        5: "B-group",
        6: "I-group",
        7: "B-location",
        8: "I-location",
        9: "B-person",
        10: "I-person",
        11: "B-product",
        12: "I-product",
        }
    
    def ids_to_labels(all_ids):
        return [
            [id2labels[id] for id in ids]
            for ids in all_ids
        ]
   

    X_train = dataset["train"]["tokens"]
    y_train = ids_to_labels(dataset["train"]["ner_tags"])
    X_test = dataset["test"][:-1]["tokens"]
    y_test = ids_to_labels(dataset["test"][:-1]["ner_tags"])
    X_val = dataset["validation"]["tokens"]
    y_val = ids_to_labels(dataset["validation"]["ner_tags"])

    # label_list = ["null", "corporation", "creative-work", "group", "location", "person", "product"]
    label_list = list(id2labels.values())
    splits = {
        "train" : { "sentences" : X_train, "labels" : y_train },
        "test" : { "sentences" : X_test, "labels" : y_test },
        "val" : { "sentences" : X_val, "labels" : y_val },
        "label_list" : label_list
    }
    
    with open(args.save_to, "wb") as f:
        torch.save(splits, f)
    
    print("Labels detected: ", label_list)
    print("Saved splits to pt:")
    print("Train", len(splits["train"]["sentences"]))
    print("Test", len(splits["test"]["sentences"]))
    print("Val", len(splits["val"]["sentences"]))

if __name__ == "__main__":
    main()
    # Usage:
    # python make_splits.py --save_to original_splits.pt
