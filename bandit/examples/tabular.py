import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import torch.nn as nn

from tqdm import tqdm
import sys
sys.path.append('.')
from examples.utils import set_random_seed, TRAINERS

class TabularPolicy(nn.Module):
    def __init__(self, num_envs, num_actions):
        super(TabularPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thetas = nn.Parameter(torch.zeros((num_envs, num_actions), device=self.device))

    def forward(self):
        return self.thetas

def plot(experiments, max_points = 1000):
    experiment = experiments[0]

    cmap = plt.get_cmap('tab10')
    num_colors = 10
    colors = cmap(np.linspace(0, 1, num_colors))
    desc_color_map = {}
    color_index = 0
    
    for key in ["Param Diff", "Value", "Value Diff"]:
        fig = plt.figure(figsize=(8, 6))
        for desc, data in experiment.items():
            base_desc = desc.replace('empirical', '').replace('exact', '')
            if "*" in base_desc: base_desc = base_desc.replace("*", "")
            if base_desc not in desc_color_map:
                desc_color_map[base_desc] = colors[color_index % num_colors]
                color_index += 1
            color = desc_color_map[base_desc]

            values = data[key]
            epochs = np.arange(1, len(values) + 1)

            if values.shape[0] > max_points:
                idx = np.linspace(0, values.shape[0] - 1, max_points).astype(int)
                epochs = epochs[idx]
                values = values[idx]

            plt.plot(epochs, values, label=desc, color=color, linestyle="-" if "empirical" in desc else "--", lw=1 if "empirical" in desc else 3)
    
        plt.xlabel('# Updates: $t$', fontsize=15)
        plt.ylabel(key, fontsize=15)
        plt.legend(fontsize=15)
        if "Diff" in key:
            plt.yscale('log')
            
        plt.tight_layout()
        fig.savefig(f'tabular_DPO_{key.lower().replace(" ", "_")}.pdf')


def multi_plot(experiments, max_points = 1000):
    num_experiments = len(experiments)
    ncols = 2
    nrows = num_experiments // ncols + (num_experiments % ncols > 0)

    cmap = plt.get_cmap('tab10')
    num_colors = 10
    colors = cmap(np.linspace(0, 1, num_colors))
    desc_color_map = {}
    color_index = 0
    
    for key in ["Param Diff", "Value", "Value Diff"]:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6 * nrows))
        axs = axs.flatten()
        for i, experiment in enumerate(experiments):
            for desc, data in experiment.items():
                base_desc = desc.replace('empirical', '').replace('exact', '')
                if "*" in base_desc: base_desc = base_desc.replace("*", "")
                if base_desc not in desc_color_map:
                    desc_color_map[base_desc] = colors[color_index % num_colors]
                    color_index += 1
                color = desc_color_map[base_desc]

                values = data[key]
                epochs = np.arange(1, len(values) + 1)

                if values.shape[0] > max_points:
                    idx = np.linspace(0, values.shape[0] - 1, max_points).astype(int)
                    epochs = epochs[idx]
                    values = values[idx]

                axs[i].plot(epochs, values, label=desc, color=color, linestyle="-" if "empirical" in desc else "--")
        
            axs[i].set_xlabel('# Updates: $t$', fontsize=15)
            axs[i].set_ylabel(key, fontsize=15)
            axs[i].legend(fontsize=15)
            if "Diff" in key:
                axs[i].set_yscale('log')

        for i in range(len(experiments), len(axs)):
            fig.delaxes(axs[i])
            
        plt.tight_layout()
        plt.savefig(f'tabular_DPO_{key.lower().replace(" ", "_")}.pdf')
        plt.close(fig)

# Hyperparameters
# Our configurations are as below:
# exact: num_iter=100, beta=3, num_actions=20, lrs=[10]
# empirical: num_iter=3000, beta=3, num_actions=20, lrs=[0.05]
exp_type = [False] # True for empirical, False for exact
num_iter = 100
beta = 3
num_actions=20
lrs = [10]

# Plotting parameters
num_envs = 1 # the number of prompts
num_samples = 8192 # the number of samples, to control the scale of noise

# set_random_seed(2225393)
set_random_seed(1234567)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rewards = torch.randn(num_envs, num_actions, device=device)
reward_margin = torch.max(rewards).cpu().numpy() - torch.min(rewards).cpu().numpy()

opt_probs = torch.softmax(rewards / beta, dim=-1)
opt_log_probs = torch.log_softmax(rewards / beta, dim=-1)
opt_values = torch.sum((rewards - beta * opt_log_probs) * opt_probs, dim=-1)

experiments = [{} for _ in range(num_envs)]


for lr in lrs:
    for empirical in exp_type:
        for name, func in TRAINERS.items():
            desc = f"{name} ({'empirical' if empirical else 'exact'})"
            print(f"Running {desc}...")
            
            if empirical == True and name == 'DPO-Mix-P': continue
            if empirical == False and name == 'DPO-Mix-P*': continue

            policy = TabularPolicy(num_envs, num_actions).to(device)
            # DPO loss function takes mean over batch, but we know our params are disjoint, so multiply it back
            optimizer = torch.optim.SGD(policy.parameters(), lr=lr*num_envs)

            param_diffs = torch.zeros((num_envs, num_iter), device=device)
            values = torch.zeros((num_envs, num_iter), device=device)

            for i in tqdm(range(num_iter)):
                optimizer.zero_grad()

                logits = policy()
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                bs = probs.shape[0]
                num_labels = probs.shape[1]
                
                A = torch.sigmoid(rewards.view(bs, -1, 1) - rewards.view(bs, 1, -1))
                B = torch.sigmoid(- rewards.view(bs, -1, 1) + rewards.view(bs, 1, -1))
                C = 1/(A * B)
                Z = torch.sum(C, dim=(1,2))
                C = (C.view(bs, num_labels*num_labels, 1) / Z.view(bs, 1, 1)).view(bs, num_labels, num_labels)
                joint_opt_probs = C
                
                A = torch.sigmoid(beta * log_probs.view(bs, -1, 1) - beta * log_probs.view(bs, 1, -1))
                B = torch.sigmoid(- beta * log_probs.view(bs, -1, 1) + beta * log_probs.view(bs, 1, -1))
                C = 1/(A * B)
                Z = torch.sum(C, dim=(1,2))
                C = (C.view(bs, num_labels*num_labels, 1) / Z.view(bs, 1, 1)).view(bs, num_labels, num_labels)
                joint_probs = C.detach()
                
                A = torch.sigmoid(beta * log_probs.view(bs, -1, 1) - beta * log_probs.view(bs, 1, -1))
                B = torch.sigmoid(- beta * log_probs.view(bs, -1, 1) + beta * log_probs.view(bs, 1, -1))
                C = 1/(A * B)
                C = torch.minimum(C, torch.ones_like(C) * (2+np.exp(reward_margin)+np.exp(-reward_margin)))
                Z = torch.sum(C, dim=(1,2))
                C = (C.view(bs, num_labels*num_labels, 1) / Z.view(bs, 1, 1)).view(bs, num_labels, num_labels)
                joint_probs_rs = C.detach()

                values[:, i] = torch.sum((rewards - beta * log_probs.detach()) * probs.detach(), dim=-1)
                v1 = (rewards - beta * log_probs.detach()).view(bs, num_actions)
                v1 = v1.view(bs, num_actions, 1).expand(bs, num_actions, num_actions)
                v2 = v1.permute(0, 2, 1)
                param_diffs[:, i] = torch.sum(torch.abs(v1 - v2)/(num_actions**2), dim=(1, 2))
                
                # This is for ablation experiments
                if 'DPO-Mix-R' in name:
                    sampler1 = torch.softmax(rewards, dim=-1)
                    sampler2 = torch.softmax(-rewards, dim=-1)
                else:
                    sampler1 = torch.softmax(-beta*logits.detach(), dim=-1)
                    sampler2 = torch.softmax(beta*logits.detach(), dim=-1)

                loss = func(
                    logits=logits,
                    probs=probs,
                    log_probs=log_probs,
                    opt_probs=opt_probs,
                    joint_probs=joint_probs,
                    joint_probs_rs=joint_probs_rs,
                    joint_opt_probs=joint_opt_probs,
                    sampler1=sampler1,
                    sampler2=sampler2,
                    num_samples=num_samples,
                    rewards=rewards,
                    beta=beta,
                    empirical=empirical,
                )
                loss.backward()
                optimizer.step()

            for e in range(num_envs):
                experiments[e][desc] = {
                    "Param Diff": np.array(param_diffs[e].cpu()),
                    "Value": np.array(values[e].cpu()),
                    "Value Diff": np.array((opt_values[e] - values[e]).cpu())
                }
            
            if num_envs == 1: plot(experiments)
            else: multi_plot(experiments)
