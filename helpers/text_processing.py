
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


def extract_frame(tokens, labels):
    frame = {}
    for token, label in zip(tokens, labels):
        if label != "null":
            try:
                frame[label].append(token)
            except:
                frame[label] = [token]
    return frame


def create_analysis_prompt(entity_set, exemplar1, exemplar2, exemplar3, exemplar4):
    try:
        entities = '\n'.join(entity_set.remove('null'))
    except:
        entities = '\n'.join(entity_set)
    return f"""You are given this set of entities:
{entities}

These entities are used as semantic tags in an XML-style.

Example 1: {exemplar1}

Example 2: {exemplar2}

Example 3: {exemplar3}

Example 4: {exemplar4}

For each entity type give a detailed definition and list the words that belong to this entity.
Format your answer as a python dict in the form of
{{ 
    entity1 : "Definition..., Words...",
    ...
}}
using the actual entities as keys.

Entity analysis:
"""
