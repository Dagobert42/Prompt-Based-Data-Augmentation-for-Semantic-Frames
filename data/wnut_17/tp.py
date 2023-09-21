import re


def get_descriptions(entity_set, entity_descriptions):
    description_text = """"""
    for e in entity_set:
        if e != "null":
            description_text += f"{str(e)}: {str(entity_descriptions[e])}\n"
            
    return description_text


def tag_exemplar(sentence, labels):
    sample = ""
    # TODO: let user define null label
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
    openings = 0
    closings = 0
    words = add_spaces(sentence).split()
    current_label = "null"
    for word in words:
        # check for label
        if word[0] == "<" and word[-1] == ">":
            if word[1] == "/":
                current_label = "null"
                closings += 1
            else:
                current_label = word[1:-1]
                openings += 1
        else: # store tokens
            if current_label not in label_list:
                raise ValueError(f"Wrong entity detected in augmentation: {current_label}")
            tokens.append(word)
            labels.append(current_label)
    if openings != closings:
        raise ValueError(f"Uneven amount of opened/closed tags ({openings} : {closings} closed) in: {sentence}")
    return tokens, labels


def extract_augmentations(responses, label_list):
    regex = '^[Ee]xample\s([0-9]|[1-9][0-9])\:\s'
    aug_sentences = []
    aug_labels = []
    errors = 0
    for response in responses:
        for paragraph in response.split('\n'):
            paragraph = re.sub(regex, '', paragraph.rstrip())
            if len(paragraph) > 5:
                try:
                    tokens, labels = parse_markup(paragraph, label_list)
                    if len(tokens) > 0:
                        aug_sentences.append(tokens)
                        aug_labels.append(labels)
                except Exception as e:
                    errors += 1
                    print(f"Error {errors}: {e}")
                    continue
            else:
                continue
    return aug_sentences, aug_labels


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


def create_context_prompt(exemplar1, exemplar2, entity_set):
    try:
        entities = '\n'.join(entity_set.remove('null'))
    except:
        entities = '\n'.join(entity_set)
    return f"""You are given this set of entities:
{entities}

These entities are used as semantic tags in an XML-style.

Sample text:
{exemplar1}
{exemplar2}

Give a concise definition for each entity and list examples that belong to this entity.
Do not add new entities.
Format your answer as a JSON dict.
"""


def create_continuation_prompt(exemplar1, entity_set):
    # TODO:
    return f"""You are given this set of entities:
{entity_set}
These entities are used exclusively as semantic tags in an XML-style.

Example 1: {exemplar1}

Come up with as many new examples as possible on the topic of the given example.
Make sure to correctly mark all entities in the new examples by surrounding them with the proper entity tags.

Example 2: 
"""


#TODO: obsolete!?
def create_context(
        descriptions,
        datapoint1,
        datapoint2,
        ):
    labels1 = datapoint1[1]
    exemplar1 = datapoint1[0]
    labels2 = datapoint2[1]
    exemplar2 = datapoint2[0]

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
