from fastapi.templating import Jinja2Templates
from typing import Optional

from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import transformers
from tqdm import tqdm
import torch
from summar import T5Dataset
from lang import MBARTDataset
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    get_linear_schedule_with_warmup
)
from transformers import AutoModelForCausalLM, AutoTokenizer


DEVICE = "cpu"
device = torch.device(DEVICE)

tokenizer_summar = T5Tokenizer.from_pretrained('Vaibhavbrkn/t5-summarization')
model_summar = T5ForConditionalGeneration.from_pretrained('t5-base')
model_summar.to(device)
model_summar = torch.quantization.quantize_dynamic(
    model_summar, {torch.nn.Linear}, dtype=torch.qint8
)


model = MBartForConditionalGeneration.from_pretrained(
    "Vaibhavbrkn/mbart-english-hindi")
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50", src_lang="en_XX", tgt_lang="hi_IN")

model.to(device)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

tokenizer_con = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model_con = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
step = 0
lst = []


def conver(inp):
    global step
    global lst
    new_user_input_ids = tokenizer_con.encode(
        inp + tokenizer_con.eos_token, return_tensors='pt')
    if step > 0:
        chat_history_ids = lst[-1]
        bot_input_ids = torch.cat(
            [chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model_con.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer_con.eos_token_id)
    lst.append(chat_history_ids)
    step += 1
    ans = tokenizer_con.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return ans


def eval(loader, model, tokenizer):
    model.eval()

    for data in loader:
        outs = model.generate(
            input_ids=data['source_ids'], attention_mask=data['src_mask'])
        with tokenizer.as_target_tokenizer():
            dec = [tokenizer.decode(ids) for ids in outs]
        texts = [tokenizer.decode(ids) for ids in data['source_ids']]
        print(data['source_ids'])

        return dec


app = FastAPI()
templates = Jinja2Templates(directory="templates/")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
    name="static",
)


@app.get("/summary")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('summarization.html', context={'request': request, 'result': result})


@app.post("/summary")
def form_post(request: Request, inp: str = Form(...)):

    article = [inp]
    dataset = T5Dataset(article, tokenizer_summar)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    ans = eval(loader, model_summar, tokenizer_summar)[0]
    ans = ans.split(" ")[1:-1]
    result = " ".join(ans)

    return templates.TemplateResponse('summarization.html', context={'request': request, 'result': result})


@app.get("/translation")
def trans_get(request: Request):
    result = ""
    return templates.TemplateResponse('translation.html', context={'request': request, 'result': result})


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


@app.post("/translation")
def trans_post(request: Request, inp: str = Form(...)):
    english = [inp]
    dataset = MBARTDataset(english, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    ans = eval(loader, model, tokenizer)[0]
    ans = ans.split(" ")[1:-1]
    result = " ".join(ans)

    return templates.TemplateResponse('translation.html', context={'request': request, 'result': result})


@app.get("/conversation")
def con_get(request: Request):
    result = ""
    return templates.TemplateResponse('conversation.html', context={'request': request, 'result': result})


@app.post("/conversation")
def con_post(request: Request, inp: str = Form(...)):
    result = conver(inp)
    return templates.TemplateResponse('conversation.html', context={'request': request, 'result': result})
