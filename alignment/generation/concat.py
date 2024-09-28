import json
import random
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    num_iter: Optional[int] = field(
        default=0,
    )
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

data_dir="data/"
iter=script_args.num_iter
res_type = ['mixp_temp1', 'mixp_temp2']
fin_type = 'mixp'
data = [None]*2
for idx in range(2):
    with open(data_dir+'data_with_rewards_iter'+str(iter)+'_'+res_type[idx]+'.json', 'r', encoding='utf-8') as f:
        data[idx] = json.load(f)
        data[idx]['instances'] = sorted(data[idx]['instances'], key=lambda x: x['prompt'])
num_prompt = len(data[0]['instances'])
for idx in range(num_prompt):
    assert data[0]['instances'][idx]['prompt'] == data[1]['instances'][idx]['prompt']
    data[1]['instances'][idx]['responses'] += data[0]['instances'][idx]['responses']
    data[1]['instances'][idx]['rewards'] += data[0]['instances'][idx]['rewards']
random.seed(iter)
random.shuffle(data[1]['instances'])
with open(data_dir+'data_with_rewards_iter'+str(iter)+'_'+fin_type+'.json', 'w', encoding='utf-8') as f:
    json.dump(data[1], f, ensure_ascii=False)