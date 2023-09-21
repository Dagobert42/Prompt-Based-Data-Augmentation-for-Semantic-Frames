import torch
from torch.utils import data

class TokenClassificationDataset(data.Dataset):
    """
    Produces tokenized pairs of sentences and labels for token classification.
    """
    def __init__(self, sentences, labels, label_dict, tokenizer, max_input_length):
            assert(len(sentences)==len(labels))
            self.sentences = sentences
            self.labels = labels
            self.label_dict = label_dict
            self.size = len(self.sentences)

            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.data = { "input_ids": [], "attention_mask": [], "labels": [] }
            self.tokenize_and_add_data(self.sentences, self.labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.data.items()}
        return item

    def __len__(self):
        return self.size
    
    def align_labels(self, tokens, labels):
        previous_word_id = None
        aligned_labels = []
        for word_id in tokens.word_ids():
            # Special tokens have a word id that is None. 
            # We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(self.label_dict[labels[word_id]])
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id
        return aligned_labels

    def tokenize_and_add_data(self, sentences, labels):
        for i, sentence in enumerate(sentences):
            tokens = self.tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=self.max_input_length,
                is_split_into_words=True
                )
            self.data["input_ids"].append(torch.reshape(tokens["input_ids"], (-1,)))
            self.data["attention_mask"].append(torch.reshape(tokens["attention_mask"], (-1,)))
            self.data["labels"].append(self.align_labels(tokens, labels[i]))
    

class SequenceClassificationDataset(data.Dataset):
    """
    Produces tokenized sentences and labels for sequence classification.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.size = len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return self.size
    