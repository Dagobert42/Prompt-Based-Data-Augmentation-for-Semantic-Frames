import re


def get_descriptions(entity_set, entity_dict):
    description_text = """"""
    for l in set(entity_set):
        if l != "null":
            description_text += l + " : " + entity_dict[l] + ",\n"
            
    return description_text


def tag_exemplar(sentence, labels):
    sample = ""
    last_label = "null"

    for word, label in zip(sentence, labels):
        if label != last_label and last_label != "null":
                sample += f"</{last_label}> "
        if label != last_label and label != "null":
                sample += f"<{label}> "
        sample += word + " "
        last_label = label

    sample = sample.strip(" . ")
    if last_label != "null":
        sample += f" </{last_label}> "
    return sample + '.'


def get_class_exemplars(seed_class, sentences, all_labels):
    # find all samples of a desired class
    samples = []
    sample_labels = []
    for sentence, labels in zip(sentences, all_labels):
        if seed_class in labels:
            samples.append(sentence)
            sample_labels.append(labels)

    return samples, sample_labels


def create_context_prompt(sentence1, labels1, sentence2, labels2):
    exemplar1 = tag_exemplar(sentence1, labels1)
    exemplar2 = tag_exemplar(sentence2, labels2)
    
    entity_set = set(labels1 + labels2)
    try:
        entities = '\n'.join(entity_set.remove('null'))
    except:
        entities = '\n'.join(entity_set)
    prompt = f"""You are given this set of entities:
{entities}

The entities are tagged in the examples in an XML-style.

Example 1: {exemplar1}

Example 2: {exemplar2}

Give a definition of each entity type and explain why the words from the examples belong to this entity.
"""
    return prompt, entity_set


def add_spaces(sentence):
    result = ''
    i = 0
    no_tag = False
    for j, char in enumerate(sentence):
        if char in '<[°':
            result += sentence[i:j] + ' '
            i = j
        elif char in '>]%':
            result += sentence[i:j+1] + ' '
            i = j + 1
        elif char in '({\\/})&$*+~#\'\":|=':
            if no_tag or char != '/':
                result += sentence[i:j] + f' {char} '
                i = j + 1
        no_tag = char != '<'
    result += sentence[i:]
    return result


def parse_markup(sentence, label_list):
    labels = []
    tokens = []
    words = add_spaces(sentence).split()
    current_label = "null"
    for word in words:
        # check for label
        if word[0] == "<" and word[-1] == ">":
            if word[1] == "/":
                current_label = "null"
            else:
                current_label = word[1:-1]
        else: # store tokens
            if current_label not in label_list:
                raise ValueError(f"Wrong entity detected in augmentation: {current_label}")
            tokens.append(word)
            labels.append(current_label)
    return tokens, labels


def extract_frame(tokens, labels):
    frame = {}
    for token, label in zip(tokens, labels):
        if label != "null":
            try:
                frame[label].append(token)
            except:
                frame[label] = [token]
    return frame


def count_sentence_appearance(all_labels, label_list):
    counts = {}
    for labels in all_labels:
        for l in label_list:
            if l in labels:
                try:
                    counts[l] += 1
                except:
                    counts[l] = 1
    return counts


def extract_augmentations(responses, label_list):
    p = 'Example \b([0-9]|[1-9][0-9])\b\:\s'
    aug_sentences = []
    aug_labels = []
    for response in responses:
        replies = [
            re.sub(p, '', n)
            for m in response["choices"]
            for n in m["message"]["content"].split(".\n")
            if n != ''
            ]
        for reply in replies:
            try:
                tokens, labels = parse_markup(reply, label_list)
                aug_sentences.append(tokens)
                aug_labels.append(labels)
            except Exception as e:
                print("Parsing the reply failed with Exception:", e)
                continue
    return aug_sentences, aug_labels


def create_context(
        descriptions,
        exemplar_1,
        exemplar_2,
        ):
    labels1 = exemplar_1[1]
    exemplar1 = tag_exemplar(exemplar_1[0], labels1)
    labels2 = exemplar_2[1]
    exemplar2 = tag_exemplar(exemplar_2[0], labels2)

    entities = set(labels1 + labels2)
    try:
        entities.remove('null')
    except:
        pass

    entity_dict = get_descriptions(entities, descriptions)

    return create_context_prompt(
        exemplar1,
        labels1,
        exemplar2,
        labels2,
        ), entity_dict
