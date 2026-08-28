"""Micro-benchmark: per-step cost of Qwen2.5-3B + LoRA next-POI training.
Measures s/step for {grad_ckpt on/off} x {batch 4/8} to find the bottleneck.
"""
import os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen2.5-3B-Instruct")
N = 5000

torch.manual_seed(42)
np.random.seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([f"<POI_{i}>" for i in range(N)])

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
model.resize_token_embeddings(len(tokenizer))
model.get_input_embeddings().weight.requires_grad_(True)

lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM", bias="none")
model = get_peft_model(model, lora_cfg)


def bench(with_ckpt, batch, steps=8):
    if with_ckpt:
        model.gradient_checkpointing_enable()
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    model.train()
    ids = torch.randint(0, len(tokenizer), (batch, 64)).cuda()
    lab = ids.clone()
    lab[:, :-1] = ids[:, 1:]
    lab[:, -1] = -100
    msk = torch.ones(batch, 64).cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(2):  # warmup
        out = model(input_ids=ids, attention_mask=msk, labels=lab)
        (out.loss / 4).backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        out = model(input_ids=ids, attention_mask=msk, labels=lab)
        (out.loss / 4).backward()
        opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


for ckpt in (False, True):
    dt = bench(ckpt, 4, steps=8)
    print(f"grad_ckpt={int(ckpt)} batch=4 -> {dt:.2f} s/step (micro)", flush=True)
dt = bench(False, 8, steps=8)
print(f"grad_ckpt=0 batch=8 -> {dt:.2f} s/step (micro)", flush=True)
print("DONE", flush=True)
