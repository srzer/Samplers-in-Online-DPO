import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="iter2_K64.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="iter2_K64_Mreward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
reward_model = script_args.reward_name_or_path
rm_pipe = AutoModelForCausalLM.from_pretrained(reward_model,
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model, use_fast=True)
tokenizer_plain = AutoTokenizer.from_pretrained(reward_model, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"

ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

"""
We process the data format here and query the reward model to get the rewards.
"""


def get_reward(test_texts):
    assert len(test_texts["responses"]) >= 2
    rm_pipe.eval()
    prefix = ["BEGINNING OF CONVERSATION: USER: ", "<|system|> You are a helpful assistant.<|end|><|user|> ", "Q: "]
    suffix = [" ASSISTANT:", "<|end|><|assistant|> ", "\nA:"]
    flg = False
    for idx in range(0, len(prefix)):
        if test_texts["prompt"].startswith(prefix[idx]) and test_texts["prompt"].endswith(suffix[idx]):
            test_texts["prompt"] = test_texts["prompt"][len(prefix[idx]):-len(suffix[idx])]
            flg = True
    if flg == False:
        raise ValueError("Invalid prompt format")
    token_id_A = rm_tokenizer.encode("A", add_special_tokens=False)
    token_id_B = rm_tokenizer.encode("B", add_special_tokens=False)
    assert len(token_id_A) == 1 and len(token_id_B) == 1
    token_id_A = token_id_A[0]
    token_id_B = token_id_B[0]
    temperature = 1.0

    ## We can also handle multi-turn conversation.
    instruction = [{"role": "user", "content": test_texts["prompt"]}
    ]
    context = tokenizer_plain.apply_chat_template(instruction, tokenize=False)
    responses = [test_texts["responses"][0], test_texts["responses"][1]]
    probs_chosen = []
        
    for chosen_position in [0, 1]:
        # we swap order to mitigate position bias
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]

        input_ids = rm_tokenizer.encode(rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, ""), return_tensors='pt', add_special_tokens=False).cuda() 

        with torch.no_grad():
            output = rm_pipe(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        # take softmax to get the probability; using numpy
        Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / temperature) / Z
        probs_chosen.append(prob_chosen)

    avg_prob_chosen = np.mean(probs_chosen)
    correct = 0.5 if avg_prob_chosen == 0.5 else float(avg_prob_chosen > 0.5)
    return [correct, 1-correct]

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < 2:
            continue
        test_texts = {
            "prompt": sample["prompt"],
            "responses": [tmp_output.strip() for tmp_output in sample["responses"]
        ]}
        rewards = get_reward(test_texts)
        data.append({"prompt": sample["prompt"], "responses": sample["responses"], "rewards": rewards})


# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

# import torch.distributed as dist

# dist.all_gather_object(all_process_list, data_to_send)
all_process_list = [data_to_send]
gathered_data = []


for i in range(world_size):
    # print(all_process_list)
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

all_rewards = [sample["rewards"] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


if local_rank == 0:
    print(
        "Collect {} data from {} inputs. mean score {} top1 score: {}".format(
            len(gathered_data), data_size, mean_scores, top1_scores
        )
    )
    if len(gathered_data) < data_size:
        print(
            "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
                script_args.K
            )
        )
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

    if script_args.record_dir is not None:
        with open(script_args.record_dir, "a") as f:
            f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")
