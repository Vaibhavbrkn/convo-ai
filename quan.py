from transformers import (
    AdamW,
    MBartForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import transformers
from tqdm import tqdm
import torch

MODEL_PATH_SUMMAR = "lang_4.bin"


model_summar = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50")
model_summar.load_state_dict(torch.load(MODEL_PATH_SUMMAR))

quantized_model = torch.quantization.quantize_dynamic(
    model_summar, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "lang_quant.bin")
# quantized_model.save_pretrained("quant_summar.bin")
