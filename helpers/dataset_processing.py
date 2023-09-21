from helpers.text_processing import *


def get_exemplars_list(splits):
    # split all paragraphs in the training data into sentences
    return [
        exemplar + "."
        for exemplar in [
            tag_exemplar(s,l) for s, l in zip(
                splits["train"]["sentences"],
                splits["train"]["labels"],
                ) if set(l) != {"null"}
                ]
        ]


def get_exemplars_per_class(splits):
    # split all paragraphs in the training data into sentences
    # then group these sentences by entity
    return {
        entity : [
            exemplar.strip() + "."
            for exemplar in [
                tag_exemplar(s,l) for s, l in zip(
                    splits["train"]["sentences"],
                    splits["train"]["labels"],
                    ) if entity in l
                    ]
                ]
        # make sure the first value in label_list is the null label
        for entity in splits["label_list"][1:]
    }


def count_label_appearance(all_labels, label_list):
    counts = {}
    for labels in all_labels:
        for l in label_list:
            if l in labels:
                try:
                    counts[l] += 1
                except:
                    counts[l] = 1
    return counts
