import json
import string
from typing import List, Dict

import fire
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer


def read_json(input_path: str) -> List[Dict]:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def write_json(data: List[Dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def init_tokenizer(tokenizer_name: str) -> LlamaTokenizer:
    llm_tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        truncation_side="left",
        legacy=False,
    )
    llm_tokenizer.pad_token = "$$"
    return llm_tokenizer


def get_char_token_map(tokenized_text: List[str], text:str) -> np.ndarray:
    char_token_map = -np.ones((len(text)), dtype=np.int32)
    orig_str = text

    start_idx = 0
    for i, token in enumerate(tokenized_text):
        if token == "$$":
            break
        # if len(token) == 0:
        #     continue
        
        while len(token) > 0:
            index = orig_str.find(token)
            if index != -1:
                char_token_map[start_idx+index:start_idx + index + len(token)] = i
                orig_str = orig_str[index+len(token):]
                start_idx += index + len(token)
                break
            else:
                token = token[1:]

        # if len(token) == 0:
        #     raise ValueError(f"{i}th Token: {token}, Tokenized text: {tokenized_text}, Text: {text}")

    return char_token_map


def get_token_range(start_index: int, end_index: int, char_token_map: np.ndarray) -> List[int]:
    sliced = char_token_map[start_index:end_index]  # exclusive
    unique = np.unique(sliced)
    unique = unique[unique != -1].tolist()
    return unique


def get_char_range(word_index: int, char_word_map: np.ndarray) -> List[int]:
    return np.where(char_word_map == word_index)[0].tolist()


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def postprocess(sample: Dict, tokenizer: LlamaTokenizer) -> Dict:
    text = sample["hallucinated_text"]
    annotations = sample["annotations"]

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    tokenized_text = [tokenizer.decode(t) for t in tokens["input_ids"][0].tolist() if t not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]]
    # tokenized_text = [tokenizer.decode(t) for t in tokens["input_ids"][0].tolist() if t not in [tokenizer.pad_token_id]] + [tokenizer.eos_token]
    char_token_map = get_char_token_map(tokenized_text, text)

    sample["tokenized_text"] = tokenized_text

    try:
        for h_type in annotations.values():
            for h_list in h_type:
                for k in h_list.keys():
                    hallucinated_answer = h_list[k]
                    char_index = list(map(int, hallucinated_answer["char_index"].split(":")))

                    if len(char_index) == 1:
                        start_index = char_index[0]
                        end_index = char_index[0] + 1
                    elif len(char_index) == 2:
                        start_index = char_index[0]
                        end_index = char_index[1]

                    hallucinated_tokens = get_token_range(start_index, end_index, char_token_map)
                
                    hallucinated_text = hallucinated_answer["name"].replace(" ", "")
                    hallucinated_text_restored = "".join([tokenized_text[i] for i in hallucinated_tokens]).replace(" ", "")
                    if remove_punctuation(hallucinated_text) != remove_punctuation(hallucinated_text_restored):
                        raise ValueError(
                            f"[0]: {hallucinated_text}\n"
                            f"[1]: {hallucinated_text_restored}\n"
                            f"[2]: {text[start_index:end_index].replace(' ', '')}\n"
                            f"[hallucinated_tokens]: {hallucinated_tokens}\n"
                            f"[tokenized_text]: {tokenized_text}\n"
                        )

                    if len(hallucinated_tokens) == 1:
                        h_list[k]["token_index"] = str(hallucinated_tokens[0])
                    else:
                        h_list[k]["token_index"] = f"{hallucinated_tokens[0]}:{hallucinated_tokens[-1]+1}"
    except Exception as e:
        print(f"Error: {e}")
        raise e

    sample["annotations"] = annotations

    return sample


def main(
    input_path: str = "extractive_test.json",
    tokenizer_name: str = "models/vicuna-7b",
) -> None:
    data = read_json(input_path)
    tokenizer = init_tokenizer(tokenizer_name)

    processed_data = []
    for sample in tqdm(data):
        processed_sample = postprocess(sample, tokenizer)
        if processed_sample is not None:
            processed_data.append(processed_sample)

    output_path = input_path.replace(".json", "_minigpt4_postprocessed.json")
    write_json(processed_data, output_path)


if __name__ == "__main__":
    fire.Fire(main)
