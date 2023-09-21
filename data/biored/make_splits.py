import argparse
import torch
from random import shuffle

def align_labels(text, annotations, initial_offset):
    words = []
    labels = []
    word_end = 0
    for a in annotations:
        assert len(a['locations']) == 1

        # add words in-between annotations
        word_start = a['locations'][0]['offset'] - initial_offset
        word_chunk = text[word_end:word_start].split()
        words += word_chunk
        labels += ['null']*len(word_chunk)

        # add words with non-null annotations
        word_end = word_start+a['locations'][0]['length']
        word_chunk = text[word_start:word_end].split()
        words += word_chunk
        labels += [a['infons']['type']]*len(word_chunk)

    # add words from end of passage
    word_chunk = text[word_end:].split()
    words += word_chunk
    labels += ['null']*len(word_chunk)
    assert (len(words) == len(labels))
    return words, labels

# TODO: let user define null label
def load_data(path):
    import json
    with open(path, 'rb') as f:
        data = json.load(f)

    passages = []
    all_labels = []
    for d in data['documents']:
        for p in d['passages']:
                words, labels = align_labels(
                    p['text'],
                    p['annotations'],
                    p['offset']
                    )
                passages.append(words)
                all_labels.append(labels)
        
    label_list = ['null']
    for labels in all_labels:
        for label in labels: # sentence level
            if label not in label_list:
                label_list.append(label)

    return passages, all_labels, label_list


def main():
    parser = argparse.ArgumentParser(description="Create BioRED data splits")
    parser.add_argument("--train_size", type=int, required=False, help="Target size of train split, potential leftovers will be added to test split", default=200)
    parser.add_argument("--save_to", required=True, help="Name or path of the .pt file to save the dataset splits to")
    args = parser.parse_args()

    X_train, y_train, label_list = load_data("Train.BioC.JSON")
    # randomly select a number of examples from train
    X = [(x, y) for x, y in zip(X_train, y_train)]
    shuffle(X)
    X_train, y_train = list(zip(*X[:args.train_size]))

    X_test, y_test, _ = load_data("Test.BioC.JSON")
    # add the left over training data to the test set
    X_rest, y_rest = list(zip(*X[args.train_size:]))
    X_test += X_rest
    y_test += y_rest

    X_val, y_val, _ = load_data("Dev.BioC.JSON")

    splits = {
        # we exchange train and test sets intentionally here to have a smaller training set
        'train' : { 'sentences' : X_train, 'labels' : y_train },
        'test' : { 'sentences' : X_test, 'labels' : y_test },
        'val' : { 'sentences' : X_val, 'labels' : y_val },
        'label_list' : label_list
    }
    
    with open(args.save_to, 'wb') as f:
        torch.save(splits, f)
    
    print("Labels detected: ", label_list)
    print("Saved splits to pt:")
    print("Train", len(splits['train']['sentences']))
    print("Test", len(splits['test']['sentences']))
    print("Val", len(splits['val']['sentences']))


if __name__ == "__main__":
    main()
    # Usage example:
    # python make_splits.py --save_to original_splits.pt --train_size 800
