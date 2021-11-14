#!/usr/bin/env python3
from typing import List, Tuple

import torch
from transformers import BertForMaskedLM, BertTokenizer

_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def print_result(words: List[str], predictions: torch.Tensor) -> None:
    for pos in range(len(words)):
        _, indexes = torch.topk(predictions[0, pos], k=5)
        tokens = _tokenizer.convert_ids_to_tokens(indexes.tolist())
        word = words[pos]
        print(f"#{pos}: {word}")
        for rank, index in enumerate(indexes):
            print(
                "- {} (#{}: {:.3f})".format(
                    tokens[rank], index.item(), predictions[0, pos, rank].item()
                )
            )
        print()


def process_sentence(text: str) -> Tuple[List[str], torch.Tensor]:
    words = _tokenizer.tokenize(text)
    words.insert(0, "[CLS]")
    words.append("[SEP]")

    for i in range(len(words)):
        if words[i] == "*":
            words[i] = "[MASK]"

    print(" ".join(words))

    tokens = _tokenizer.convert_tokens_to_ids(words)
    input_tensor = torch.tensor([tokens])

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = outputs[0]

    return words, predictions


if __name__ == "__main__":
    while True:
        text = input("TEXT> ").lower()
        if not text:
            break
        words, predictions = process_sentence(text)
        print_result(words, predictions)
