import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from random import shuffle
from tqdm import tqdm
from helpers.text_processing import *
from helpers.dataset_processing import *
import time

def main():
    parser = argparse.ArgumentParser(description="Data Analysis Script")

    parser.add_argument("--save_to", required=False, help="Path for output", default="")
    parser.add_argument("--data_file", required=True, help="Path to data")
    parser.add_argument("--gpu_ids", type=str, required=False, help="Specifies the GPUs to use for training", default="-1")
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

    class_exemplars = get_exemplars_per_class(splits)
    # shuffle to diversify seed generation
    for key in class_exemplars.keys():
        shuffle(class_exemplars[key])

    responses = []
    seen_entities = set()
    for entity in tqdm(splits["label_list"][1:]):
        try:
            _, labels = parse_markup(" ".join(class_exemplars[entity][:4]), splits["label_list"])
        except:
            print(f"Not enough exemplars for class label: {entity}")
            print(f"Try analysing this class manually:\n{' '.join(class_exemplars[entity])}")
            continue
        entity_set = set(labels)
        analysis_prompt = create_analysis_prompt(
            entity_set,
            *class_exemplars[entity][:4]
            )
        print(analysis_prompt)

        seen_entities |= entity_set
        messages=[{"role" : "user", "content" : analysis_prompt}]
        encodings = tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
            )
        model_inputs = encodings.to(device)
        generated_ids = model.generate(
            model_inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=4096,
            do_sample=True
            )
        decoded = tokenizer.batch_decode(generated_ids)[0]
        responses.append(decoded[decoded.rfind("Entity analysis:")+16:decoded.rfind("</s>")])
        print(responses[-1])
        if seen_entities == set(splits["label_list"]):
            break

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(args.save_to+timestamp+"_entity_analysis.txt", "w") as f:
        f.write("\n\n".join(responses))

if __name__ == "__main__":
    main()
