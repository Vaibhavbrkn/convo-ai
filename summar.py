from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import transformers
from tqdm import tqdm
import torch

MODEL_PATH = "summar.bin"
DEVICE = "cpu"
device = torch.device(DEVICE)
MAX_LEN = 512


class T5Dataset:
    def __init__(self, review, tokenizer):
        self.review = review
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        source_tokenizer = self.tokenizer.encode_plus(
            review, max_length=MAX_LEN, pad_to_max_length=True, return_tensors="pt")

        source_ids = source_tokenizer["input_ids"].squeeze()
        src_mask = source_tokenizer["attention_mask"].squeeze()

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long),
        }
