from re import sub
import pandas as pd
from itertools import cycle
import time
import os
import torch
from helpers.text_processing import *


def extract_augmentations(responses, label_list):
    regex = '^[Ee]xample\s([0-9]|[1-9][0-9])\:\s'
    aug_sentences = []
    aug_labels = []
    errors = 0
    for response in responses:
        for paragraph in response.split('.\n'):
            paragraph = sub(regex, '', paragraph.rstrip())
            try:
                tokens, labels = parse_markup(paragraph, label_list)
                if len(tokens) > 0:
                    aug_sentences.append(tokens)
                    aug_labels.append(labels)
            except Exception as e:
                errors += 1
                print(f"Error {errors}: {e}")
                continue
    return aug_sentences, aug_labels


# TODO: this is for gippidy3.5 responses
# def extract_augmentations(responses, label_list):
#     p = '^[Ee]xample\s([0-9]|[1-9][0-9])\:\s'
#     aug_sentences = []
#     aug_labels = []
#     for response in responses:
#         replies = [
#             re.sub(p, '', n)
#             for m in response["choices"]
#             for n in m["message"]["content"].split(".\n")
#             if n != ''
#             ]
#         for reply in replies:
#             try:
#                 tokens, labels = parse_markup(reply, label_list)
#                 aug_sentences.append(tokens)
#                 aug_labels.append(labels)
#             except Exception as e:
#                 print("Parsing the reply failed with Exception:", e)
#                 continue
#     return aug_sentences, aug_labels


def calculate_distance_to_top(all_labels, label_list, factor=100):
    # calculates for each class the difference in percent between
    # its own number of representations and the top represented class
    label_counts = pd.Series(
        [l for labels in all_labels for l in labels]
        ).value_counts()
    try:
        counts = label_counts[label_list]
    except KeyError as e:
        print("ERROR: One or more classes were not found in the labels\n", label_counts, e)
        return
    # if the most represented class was featured 100 times
    # we return 0 for that class and for each other class
    # we return the percentage of times it would have had to
    # be added to the dataset in order to be featured 100 times
    counts = (((counts - counts.max()) * -1) * factor / counts.max()).astype(int)
    counts['total'] = counts.sum()
    return 


def cycle_exemplars(exemplars, n):
    i = 0
    while True:
        yield [exemplars[(i + j) % len(exemplars)] for j in range(n)]
        i = (i + 1) % len(exemplars)

def exemplar_selector(seed_class, exemplars_per_class, all_exemplars):
    all_cycle = cycle_exemplars(all_exemplars, 3)
    class_cycle = cycle(exemplars_per_class[seed_class])
    while True:
        yield [next(class_cycle)] + next(all_cycle)


def save_all(save_to, responses, all_labels):
    try:
        os.makedirs(save_to)
    except FileExistsError:
        pass
    file_id = time.strftime("%Y%m%d-%H%M%S")
    with open(save_to+file_id+"_responses.pt", 'wb') as f:
        torch.save(responses, f)
    with open(save_to+file_id+"_used_labels.pt", 'wb') as f:
        torch.save(all_labels, f)

def create_augmented_splits(responses, splits, save_to):
    aug_sentences, aug_labels = extract_augmentations(responses, splits["label_list"])
    splits["train"]["sentences"] = aug_sentences
    splits["train"]["labels"] = aug_labels

    with open(save_to+"augmented_splits.pt", "wb") as f:
        torch.save(splits, f)
