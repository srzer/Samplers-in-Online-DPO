import base64
import copy
import hashlib
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
import torch
import torch.nn as nn
import torch.optim as optim

from scipy import stats
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

def set_random_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU and CUDA
    torch.cuda.manual_seed(seed)  # CUDA
    torch.cuda.manual_seed_all(seed)  # all GPUs, if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash build
    
    # The following lines ensure further determinism. However, they might reduce the performance.
    # It's important to test and decide if they are acceptable for your use case.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hash_dict(d):
    def is_transform_method(x):
        for name in dir(transforms):
            obj = getattr(transforms, name)
            if inspect.isclass(obj) and isinstance(x, obj):
                return True
        return False

    def transform_to_str(transform):
        if hasattr(transform, 'transforms'):
            return ';'.join([transform_to_str(t) for t in transform.transforms])
        else:
            return f"{transform.__class__.__name__}:{getattr(transform, 'size', '')}"

    serializable_dict = {}
    for key, value in d.items():
        if is_transform_method(value):
            serializable_dict[key] = transform_to_str(value)
        else:
            serializable_dict[key] = value
        
    dict_str = json.dumps(serializable_dict, sort_keys=True)
    full_hash = hashlib.sha256(dict_str.encode()).digest()
    short_hash = base64.urlsafe_b64encode(full_hash)[:10].decode()

    return short_hash

class InMemoryDatasetWrapper(Dataset):
    def __init__(self, dataset_class, **dataset_kwargs):
        root = dataset_kwargs.get('root', None)
        assert root is not None, 'dataset root must be provided'

        self.dataset_dir = os.path.join(root, dataset_class.__name__, hash_dict(dataset_kwargs) + ".pt")
        os.makedirs(os.path.dirname(self.dataset_dir), exist_ok=True)

        if not os.path.exists(self.dataset_dir):
            self.data = []
            while True:
                try:
                    dataset = dataset_class(**dataset_kwargs)
                    break
                except Exception as e:
                    # Handle dataset download error
                    dataset_kwargs["download"] = False
                    continue
            print("Loading original dataset...")
            for i in tqdm(range(len(dataset))):
                self.data.append(dataset[i])
            print("Saving dataset to disk...")
            torch.save(self.data, self.dataset_dir)
            print(f"Dataset saved to {self.dataset_dir}")
        else:
            self.data = torch.load(self.dataset_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def plot_values_with_confidence(values_dict, task_name, smooth_window=49, max_points=1000):
    plt.figure(figsize=(10, 6))

    smooth_kernel = np.ones(smooth_window) / smooth_window

    for trainer_name, all_run_values in values_dict.items():
        all_run_values = np.array(all_run_values)
        
        mean_values = np.mean(all_run_values, axis=0)
        stderr = stats.sem(all_run_values, axis=0)
        confidence_interval = stderr * stats.t.ppf((1 + 0.95) / 2., len(all_run_values)-1)

        mean_values = np.convolve(mean_values, smooth_kernel, mode='valid')
        confidence_interval = np.convolve(confidence_interval, smooth_kernel, mode='valid')

        epochs = np.arange(1, len(mean_values) + 1)

        if len(mean_values) > max_points:
            idx = np.linspace(0, len(mean_values) - 1, max_points).astype(int)
            epochs = epochs[idx]
            mean_values = mean_values[idx]
            confidence_interval = confidence_interval[idx]
        
        plt.fill_between(epochs, mean_values - confidence_interval, mean_values + confidence_interval, alpha=0.2)
        plt.plot(epochs, mean_values, label=f'{trainer_name}')

    plt.xlabel('# Updates')
    plt.ylabel('Value')
    plt.title('Mean Values with 95% Confidence Interval')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name}.pdf')
    plt.close()

def supervised_learning(probs, log_probs, labels, beta, **kwargs):
    axis = torch.arange(probs.shape[0], device=probs.device)

    log_prob = log_probs[axis, labels.view(-1)]

    return - log_prob.mean() - beta * torch.sum(probs * log_probs, dim=-1).mean()

def PG(probs, log_probs, rewards, beta, **kwargs):
    return - torch.sum(rewards * probs, dim=-1).mean() - beta * torch.sum(probs * log_probs, dim=-1).mean()

def empirical_PG(probs, log_probs, rewards, actions, beta, **kwargs):
    axis = torch.arange(probs.shape[0], device=probs.device)

    log_prob = log_probs[axis, actions.view(-1)]
    reward = rewards[axis, actions.view(-1)]

    return - (reward * log_prob).mean() - beta * torch.sum(probs * log_probs, dim=-1).mean()

def DPO_inner(samp_prob_a1, samp_prob_a2, log_probs, rewards, beta, **kwargs):
    weight_matrix = samp_prob_a1.unsqueeze(2) * samp_prob_a2.unsqueeze(1)

    bs = rewards.shape[0]
    
    preferences = torch.sigmoid(rewards.view(bs, -1, 1) - rewards.view(bs, 1, -1))

    log_probs_contrast = log_probs.view(bs, -1, 1) - log_probs.view(bs, 1, -1)

    loss_item = preferences * nn.LogSigmoid()(beta * log_probs_contrast) + (1 - preferences) * nn.LogSigmoid()(-beta * log_probs_contrast)

    if torch.isinf(loss_item).any():
        pdb.set_trace()

    return - ((loss_item * weight_matrix).sum(dim=(1,2))).mean()

def empirical_DPO_inner(samp_prob_a1, samp_prob_a2, log_probs, rewards, beta, num_samples = 1, **kwargs):
    device = log_probs.device
    (bs, num_classes) = rewards.size()
    axis = torch.arange(bs, device=device).unsqueeze(-1)
    
    a1 = torch.multinomial(samp_prob_a1, num_samples, replacement=True)
    a2 = torch.multinomial(samp_prob_a2, num_samples, replacement=True)
    
    preferences = torch.sigmoid(rewards[axis, a1] - rewards[axis, a2])
    emp_pref = torch.bernoulli(preferences).long()
    # emp_pref: 1 if a1 is preferred, 0 if a2 is preferred for each sample

    log_probs_contrast = log_probs[axis.expand_as(a1), a1] - log_probs[axis.expand_as(a2), a2]
    log_probs_contrast = (2 * emp_pref - 1) * log_probs_contrast

    # For each environment
    loss = -torch.log(torch.sigmoid(beta * log_probs_contrast)).mean(dim=-1)
    return loss.mean()

def joint_sample(joint_probs, num_samples = 1):
    flatten_probs = joint_probs.reshape(joint_probs.shape[0], -1)
    sampled_indices = torch.multinomial(flatten_probs, num_samples, replacement=True)
    res = torch.tensor(np.array(divmod(sampled_indices.cpu().numpy(), joint_probs.size(1)))).to(joint_probs.device)
    return res

def empirical_joint_DPO_inner(joint_prob, log_probs, rewards, beta, num_samples = 1, **kwargs):
    device = log_probs.device
    (bs, num_classes) = rewards.size()
    axis = torch.arange(bs, device=device).unsqueeze(-1)
    
    a1, a2 = joint_sample(joint_prob, num_samples)
    
    preferences = torch.sigmoid(rewards[axis, a1] - rewards[axis, a2])
    emp_pref = torch.bernoulli(preferences).long()
    # emp_pref: 1 if a1 is preferred, 0 if a2 is preferred for each sample

    log_probs_contrast = log_probs[axis.expand_as(a1), a1] - log_probs[axis.expand_as(a2), a2]
    log_probs_contrast = (2 * emp_pref - 1) * log_probs_contrast

    # For each environment
    loss = -torch.log(torch.sigmoid(beta * log_probs_contrast)).mean(dim=-1)
    return loss.mean()    

def joint_DPO_inner(joint_prob, log_probs, rewards, beta, **kwargs):
    bs = rewards.shape[0]
    
    preferences = torch.sigmoid(rewards.view(bs, -1, 1) - rewards.view(bs, 1, -1))

    log_probs_contrast = log_probs.view(bs, -1, 1) - log_probs.view(bs, 1, -1)

    loss_item = preferences * nn.LogSigmoid()(beta * log_probs_contrast) + (1 - preferences) * nn.LogSigmoid()(-beta * log_probs_contrast)

    if torch.isinf(loss_item).any():
        pdb.set_trace()

    return - (loss_item * joint_prob).sum(dim=(1,2)).mean()

def uniform_uniform_DPO(probs, empirical, **kwargs):
    inner_func = empirical_DPO_inner if empirical else DPO_inner
    num_classes = probs.shape[1]
    return inner_func(samp_prob_a1 = torch.ones_like(probs) / num_classes,
                      samp_prob_a2 = torch.ones_like(probs) / num_classes,
                      **kwargs)

def optimal_uniform_DPO(opt_probs, empirical, **kwargs):
    inner_func = empirical_DPO_inner if empirical else DPO_inner
    num_classes = opt_probs.shape[1]
    return inner_func(samp_prob_a1 = opt_probs,
                      samp_prob_a2 = torch.ones_like(opt_probs) / num_classes,
                      **kwargs)

def joint_on_policy_DPO(joint_probs, empirical, **kwargs):
    inner_func = empirical_joint_DPO_inner if empirical else joint_DPO_inner
    return inner_func(joint_prob = joint_probs,
                      **kwargs)

def joint_optimal_DPO(joint_opt_probs, empirical, **kwargs):
    inner_func = empirical_joint_DPO_inner if empirical else joint_DPO_inner
    return inner_func(joint_prob = joint_opt_probs,
                      **kwargs)

def joint_on_policy_DPO_rs(joint_probs_rs, empirical, **kwargs):
    inner_func = empirical_joint_DPO_inner if empirical else joint_DPO_inner
    return inner_func(joint_prob = joint_probs_rs,
                      **kwargs)

def sampler_uniform_DPO(sampler1, empirical, **kwargs):
    inner_func = empirical_DPO_inner if empirical else DPO_inner
    num_classes = sampler1.shape[1]
    return inner_func(samp_prob_a1 = sampler1,
                      samp_prob_a2 = torch.ones_like(sampler1) / num_classes,
                      **kwargs)

def sampler_sampler_DPO(sampler1, sampler2, empirical, **kwargs):
    inner_func = empirical_DPO_inner if empirical else DPO_inner
    return inner_func(samp_prob_a1 = sampler1,
                      samp_prob_a2 = sampler2,
                      **kwargs)

def proposed_DPO(**kwargs):
    weights = [0.5, 0.5]
    return weights[0]*uniform_uniform_DPO(**kwargs)+weights[1]*sampler_sampler_DPO(**kwargs)
    
def mixed_optimal_DPO(**kwargs):
    rewards = kwargs["rewards"]
    beta = kwargs["beta"]
    kwargs.pop('sampler1', None)
    kwargs.pop('sampler2', None)
    opt_probs = torch.softmax(rewards, dim=-1)
    inv_probs = torch.softmax(-rewards, dim=-1)
    weights = [0.5, 0.5]
    return weights[0]*uniform_uniform_DPO(**kwargs) + weights[1]*sampler_sampler_DPO(sampler1=opt_probs, sampler2=inv_probs, **kwargs)

def mixed_on_policy_DPO(**kwargs):
    log_probs = kwargs["log_probs"]
    beta = kwargs["beta"]
    kwargs.pop('sampler1', None)
    kwargs.pop('sampler2', None)
    opt_probs = torch.softmax(log_probs*beta, dim=-1).detach()
    inv_probs = torch.softmax(-log_probs*beta, dim=-1).detach()
    weights = [0.5, 0.5]
    return weights[0]*uniform_uniform_DPO(**kwargs) + weights[1]*sampler_sampler_DPO(sampler1=opt_probs, sampler2=inv_probs, **kwargs)

def on_policy_uniform_DPO(probs, empirical, **kwargs):
    inner_func = empirical_DPO_inner if empirical else DPO_inner
    num_classes = probs.shape[1]
    return inner_func(samp_prob_a1 = probs.detach(),
                      samp_prob_a2 = torch.ones_like(probs) / num_classes,
                      **kwargs)

TRAINERS = {
    'DPO-Unif': uniform_uniform_DPO,
    # 'DPO-Mix-R(➀)': uniform_uniform_DPO,
    # 'DPO-Mix-R(➁)': sampler_sampler_DPO,
    # 'mixed_optimal_DPO': mixed_optimal_DPO,
    # 'mixed_on_policy_DPO': mixed_on_policy_DPO,
    'DPO-Mix-R': joint_optimal_DPO,
    'DPO-Mix-P': joint_on_policy_DPO,
    'DPO-Mix-P*': joint_on_policy_DPO_rs,
    # 'optimal_uniform_DPO': optimal_uniform_DPO,
    # 'on_policy_uniform_DPO': on_policy_uniform_DPO,
    # 'sampler_uniform_DPO': sampler_uniform_DPO,
}

def inner_training_loop(loss_func, model, optimizer, reward_func, trainloader, beta, num_epochs=10, logging_interval=-1, progress_callback=None):
    device = model.device

    total_steps = len(trainloader) * num_epochs
    step_counter, prev = 0, 0

    values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            rewards = reward_func(labels)

            optimizer.zero_grad()

            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1)
     
            bs = probs.shape[0]
            num_labels = probs.shape[1]
            
            A = torch.sigmoid(rewards.view(bs, -1, 1) - rewards.view(bs, 1, -1))
            B = torch.sigmoid(- rewards.view(bs, -1, 1) + rewards.view(bs, 1, -1))
            C = 1/(A * B)
            Z = torch.sum(C, dim=(1,2))
            C = (C.view(bs, num_labels*num_labels, 1) / Z.view(bs, 1, 1)).view(bs, num_labels, num_labels)
            joint_opt_probs = C
            
            A = torch.sigmoid(beta*log_probs.view(bs, -1, 1) - beta*log_probs.view(bs, 1, -1))
            B = torch.sigmoid(- beta*log_probs.view(bs, -1, 1) + beta*log_probs.view(bs, 1, -1))
            C = 1/(A * B)
            Z = torch.sum(C, dim=(1,2))
            C = (C.view(bs, num_labels*num_labels, 1) / Z.view(bs, 1, 1)).view(bs, num_labels, num_labels)
            joint_probs = C.detach()
            
            values.append(torch.sum(rewards * probs.detach(), dim=-1).mean().cpu())

            loss = loss_func(logits=logits,
                             probs=probs,
                             joint_probs=joint_probs,
                             joint_opt_probs=joint_opt_probs,
                             empirical=False,
                             log_probs=log_probs,
                             labels=labels,
                             rewards=rewards,
                             actions=actions,
                             beta=beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % logging_interval == logging_interval - 1:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / logging_interval:.3f}')
                running_loss = 0.0
            
            step_counter += 1
            if progress_callback and step_counter / total_steps >= prev + 0.01:
                prev = step_counter / total_steps
                progress_callback(step_counter / total_steps)
    
    return values

def train(task_name, base_model, reward_func, beta, num_runs, num_epochs, trainloader, testloader):
    os.makedirs('checkpoints', exist_ok=True)
    ckpt = f'checkpoints/{task_name}.pt'
    values_by_trainer = torch.load(ckpt) if os.path.exists(ckpt) else {}
    values_by_trainer = {k: v for k, v in values_by_trainer.items() if k in TRAINERS}
    for name, func in TRAINERS.items():
        print(f'Running {name}...')
        if name in values_by_trainer:
            print(f'Found data, skipping...')
            plot_values_with_confidence(values_by_trainer, task_name)
            continue

        acc, all_run_values = [], []
        pbar = tqdm(total=num_runs, bar_format='{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]')

        for run in range(num_runs):
            model = copy.deepcopy(base_model)
            optimizer = optim.Adam(model.parameters())

            def update_progress(inner_progress):
                pbar.n = run + inner_progress
                pbar.refresh()

            # Training loop
            values = inner_training_loop(
                loss_func=func,
                model=model,
                optimizer=optimizer,
                reward_func=reward_func,
                trainloader=trainloader,
                beta=beta,
                num_epochs=num_epochs,
                progress_callback=update_progress
            )
            all_run_values.append(values)

            # Evaluation loop
            acc.append(eval(model, testloader))

        pbar.close()

        values_by_trainer[name] = all_run_values
        plot_values_with_confidence(values_by_trainer, task_name)
        torch.save(values_by_trainer, ckpt)

        avg = round(sum(acc) / len(acc) * 100, 2)
        acc = [round(x * 100, 2) for x in acc]
        print(f'Accuracy over {num_runs} runs: {acc} average: {avg}%')

def eval(model, testloader):
    device = model.device

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            bs = inputs.size(0)
            axis = torch.linspace(0, bs - 1, bs).long()

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=-1)

            total += labels.size(0)
            correct += (probs[axis, labels]).sum().item()
    
    return correct / total

def language_inner_training_loop(loss_func, model, optimizer, reward_func, trainloader, beta, num_epochs=10, logging_interval=-1, progress_callback=None):
    device = model.device

    total_steps = len(trainloader) * num_epochs
    step_counter, prev = 0, 0

    values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in tqdm(enumerate(trainloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            rewards = reward_func(labels)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1)

            values.append(torch.sum(rewards * probs.detach(), dim=-1).mean().cpu())

            loss = loss_func(logits=logits,
                             probs=probs,
                             empirical=False,
                             log_probs=log_probs,
                             labels=labels,
                             rewards=rewards,
                             actions=actions,
                             beta=beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % logging_interval == logging_interval - 1:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / logging_interval:.3f}')
                running_loss = 0.0
            
            step_counter += 1
            if progress_callback and step_counter / total_steps >= prev + 0.01:
                prev = step_counter / total_steps
                progress_callback(step_counter / total_steps)
    
    return values
  

def language_eval(model, testloader):
    device = model.device

    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            bs = input_ids.size(0)
            axis = torch.linspace(0, bs - 1, bs).long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=-1)

            total += labels.size(0)
            correct += (probs[axis, labels]).sum().item()
    
    return correct / total