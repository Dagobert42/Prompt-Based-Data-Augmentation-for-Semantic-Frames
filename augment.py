import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import time
import json
from helpers.text_processing import *
from helpers.dataset_processing import *
from helpers.augmentation import *

system_prompt = """You're a social media user. Behave as human-like as possible and keep short and informal.
Share a variety of positive, negative and neutral opinions.
Occasionally produce spelling mistakes and acronyms to appear more human-like.
"""

timestamp = lambda: time.strftime("%Y%m%d-%H%M%S")

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Script")

    parser.add_argument("--data_file", type=str, required=True, help="Path to data")
    parser.add_argument("--entity_descriptions_file", type=str, required=True, help="Path to entity dictionary")
    parser.add_argument("--save_to", type=str, required=False, help="Name or path of the .pt file to save the dataset splits to", default="")
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
    parser.add_argument("--max_calls_per_class", type=int, required=False, help="Max number of inference calls to make per round", default=500)
    parser.add_argument("--rounds", type=int, required=False, help="Number of rounds to produce augmentations", default=5)
    parser.add_argument("--save_every", type=int, required=False, help="Interval of inference calls at which to autosave all augmentations", default=50)

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Number of available GPUs: {torch.cuda.device_count()} using {device}")

    tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
    model = AutoModelForCausalLM.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
    model.to(device)

    with open(args.data_file, "rb") as f:
        splits = torch.load(f)
    all_exemplars = get_exemplars_list(splits)
    sorted_exemplars = get_exemplars_per_class(splits)
    
    with open(args.entity_descriptions_file, "rb") as f:
        entity_descriptions = json.load(f)

    distances = pd.DataFrame(columns=splits["label_list"][1:]+["total"])
    distances = distances.append(
        calculate_distance_to_top(
            splits["train"]["labels"],
            splits["label_list"][1:],
            args.max_calls_per_class
            ),
        ignore_index=True
        )

    responses = []
    labels_used = []
    for i in range(args.rounds):
        print(f"Round {i}: Total inference calls to make this round: {distances.sum()}")
        print(distances)
        for j, entity in enumerate(splits["label_list"][1:]):
            seed_selector = exemplar_selector(entity, sorted_exemplars, all_exemplars)
            for j in tqdm(range(distances[entity]), f"Augmenting {entity} ({j+1}/{len(splits['label_list'][1:])})"):
                seed_exemplars = next(seed_selector)
                # try:
                #     _, labels = parse_markup(" ".join(seed_exemplars), splits["label_list"])
                # except: # some exemplars can throw exceptions due to the way they are split
                #     continue
                # labels_used.append(labels)
                entity_set = set(splits['label_list'][1:])

                analysis_prompt = create_analysis_prompt(
                    entity_set,
                    *seed_exemplars
                    )
                entity_analysis = get_descriptions(entity_set, entity_descriptions)
                augmentation_prompt = f"""Write as many new examples as possible in the style of the given examples.
Try to include entities of the type {entity}.
Make sure to correctly mark all entities in the new examples by surrounding them with the proper entity tags.

Example 5:"""
                messages=[
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : analysis_prompt},
                    {"role" : "assistant", "content" : entity_analysis},
                    {"role" : "user", "content" : augmentation_prompt},
                    ]
                print("\n".join([system_prompt, analysis_prompt, entity_analysis, augmentation_prompt]))
                encodings = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    max_length=4096,
                    )
                model_inputs = encodings.to(device)
                generated_ids = model.generate(
                    model_inputs,
                    max_new_tokens=8192,
                    do_sample=True
                    )
                decoded = tokenizer.batch_decode(generated_ids)[0]
                responses.append(decoded[decoded.rfind("Example 5:")+10:decoded.rfind("</s>")])
                print(responses[-1])
        
        # save progress after every round
        save_all(args.save_to, responses, labels_used)
        _, aug_labels = extract_augmentations(responses, splits["label_list"])
        distances = distances.append(
            calculate_distance_to_top(
                aug_labels,
                splits["label_list"],
                args.max_calls_per_class
                ),
            ignore_index=True
            )
        distances.to_csv(args.save_to+f"distances.csv")
                     
    save_all(args.save_to, responses, labels_used)
    
    create_augmented_splits(responses, splits, args.save_to)

if __name__ == "__main__":
    main()
