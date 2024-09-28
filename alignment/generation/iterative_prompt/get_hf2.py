#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
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
model = LlamaForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=script_args.flash_attention,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0")
model.config.use_cache = True
model.eval()
tokenizer = LlamaTokenizer.from_pretrained(script_args.ref_model_name_or_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

if "iterative-prompt-v1-iter1" in script_args.dataset_name_or_path:
    ds1 = load_dataset("RLHFlow/iterative-prompt-v1-iter1-20K", split="train")
    ds2 = load_dataset("RLHFlow/iterative-prompt-v1-iter2-20K", split="train")
    ds3 = load_dataset("RLHFlow/iterative-prompt-v1-iter3-20K", split="train")
    ds4 = load_dataset("RLHFlow/iterative-prompt-v1-iter4-20K", split="train")
    ds = concatenate_datasets([ds1, ds2, ds3, ds4])
else: 
    raise NotImplementedError
ds = ds.map(
    lambda x: {
          "prompt": 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n'+x["context_messages"][0]["content"]+'\n\n### Response:\n'
    }
)
ds = ds.filter(lambda example: len(example["prompt"]) <= script_args.max_input_length)
num_iter = script_args.num_iter

batch_size=40
if num_iter > 0:
    ds = ds.select(range(10000))
else:
    ds = ds.select(range(10000, 12000))

data_size = len(ds["prompt"])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
print(ds, script_args.dataset_name_or_path)
print(ds[0])

prompts = ds["prompt"]
tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=script_args.max_input_length)
dataset = TensorDataset(tokenized_prompts['input_ids'].type(torch.long).to("cuda:0"), tokenized_prompts['attention_mask'].type(torch.bool).to("cuda:0"))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
K = script_args.K
outputs = [[] for _ in range(K)]
for input_ids, attention_masks in tqdm(dataloader):
    res = model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=input_ids.shape[1]+script_args.max_new_tokens, do_sample=True, temperature=script_args.temperature, top_p=1.0, num_return_sequences=K, pad_token_id=tokenizer.pad_token_id)
    prompt_text = [None] * (res.shape[0]//K)
    for idx in range(0, res.shape[0]//K):
        prompt_text[idx] = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
    for idx in range(0, res.shape[0]//K):
        for jdx in range(0, K):
            text = tokenizer.decode(res[idx*K+jdx], skip_special_tokens=True)
            outputs[jdx].append(text[len(prompt_text[idx]):])
    
completions = []
used_prompts = []
gathered_data = []
for i in range(len(outputs[0])):
    tmp_data = {"prompt": prompts[i], "responses": [outputs[idx][i][:] for idx in range(len(outputs))]}
    gathered_data.append(tmp_data)


output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
print("I collect ", len(gathered_data), "samples")


with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
