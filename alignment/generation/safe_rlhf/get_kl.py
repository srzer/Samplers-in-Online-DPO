#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import json
import sys
sys.path.append('./generation')
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    ref_model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the reference model name or path"},
    )
    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    num_iter: Optional[int] = field(
        default=1,
        metadata={"help": "the number of iterations"},
    )
    flash_attention: Optional[bool] = field(
        default=True,
        metadata={"help": "the flash attention"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)
model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_flash_attention_2=script_args.flash_attention,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0")
model.config.use_cache = True
ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.ref_model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=script_args.flash_attention,
        trust_remote_code=True
    ).to("cuda:1")
ref_model.config.use_cache = True
ref_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

ds = load_dataset(script_args.dataset_name_or_path, split="train")
ds = ds.map(
    lambda x: {
          "prompt": 'BEGINNING OF CONVERSATION: USER: '+x["prompt"]+' ASSISTANT:'
    }
)
ds = ds.filter(lambda example: len(example["prompt"]) <= script_args.max_input_length)
num_iter = script_args.num_iter
batch_size=1
ds = ds.select(range(10000, 10200))

data_size = len(ds["prompt"])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

prompts = ds["prompt"]
tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=script_args.max_input_length)
dataset = TensorDataset(tokenized_prompts['input_ids'].type(torch.long).to("cuda:0"), tokenized_prompts['attention_mask'].type(torch.bool).to("cuda:0"))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
K = 1
outputs = [[] for _ in range(K)]
kl = []
model.eval()
ref_model.eval()
for input_ids, attention_masks in tqdm(dataloader):
    res = model.generate(input_ids=input_ids.to(model.device), attention_mask=attention_masks.to(model.device), max_length=input_ids.shape[1]+script_args.max_new_tokens, do_sample=True, temperature=script_args.temperature, top_p=1.0, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    text = tokenizer.decode(res[0], skip_special_tokens=True)
    kl_div = 0
    for i in range(input_ids.shape[-1], res.shape[-1]):
        with torch.no_grad():
            outputs = model(res[:, :i])
            ref_outputs = ref_model(res[:, :i].to(ref_model.device))
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(ref_outputs.logits, dim=-1)
        token_id = res[0, i].item()
        p = log_probs[0, -1, token_id].item()
        q = ref_log_probs[0, -1, token_id].item()
        kl_div += p-q
    kl.append(kl_div)
    
print(np.mean(kl))