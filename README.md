# The Crucial Role of Samplers in Online DPO

This repo provides code for the paper **"The Crucial Role of Samplers in Online Direct Preference Optimization"**. The `bandit` directory is to reproduce our numerical simulations, and the `alignment` directory is to reproduce our LM alignment experiments. **This repo is still under construction. If you find any issue in reproduction, feel free to create an issue!**

## Numerical simulations
The numerical simulations can be easily reproduced by running
```bash
cd bandit
python examples/tabular.py
```
The hyperparameters can be set in `example/tabular.py`. Basic environment configurations can run our code well. 

Next we will introduce how to run our LM alignment experiments.

## Set up
Our codebase is mainly based on [**RLHFlow**](https://github.com/RLHFlow/Online-RLHF), and the configurations are mostly same. *We directly borrow some instructions from that repository in this section.*

It is recommended to have three separate environments for **inference**, **training** and **evaluation**, respectively. 

**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets
# The following code is tested for CUDA12.0-12.2. You may need to update the torch and flash-attention sources according to your own CUDA version
pip3 install torch==2.1.2 torchvision torchaudio
pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp310-cp310-manylinux1_x86_64.whl 
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install accelerate==0.27.2
pip install deepspeed
```

**Training Environment**

```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2
```

**Evaluation Environment**
```sh
conda create -n rewardflow --clone rlhflow
conda activate rewardflow
pip install transformers==4.38.2
```

You also need to install the wandb to record the training and login with your huggingface account so that you have access to the LLaMA3 models.

```sh
pip install wandb

wandb login
huggingface-cli login
```

You can also directly refer to the `alignment/requirement_*.yaml`.

## Pipeline

The whole pipeline is composed of **data generation**, **data annotation** and **DPO training**, and will repeat $3$ iterations for each approach. (Since we employ off-shelf tuned models, the sft stage is omitted.) In this codebase, we denote *vanilla DPO* as `ref` or `offline`, *on-policy DPO* as `online`, *hybrid GSHF* as `gshf`, and *ours* as `mixp`.

**Training**

For training, we provide a putting-everything-together script, `sample_train_safe_rlhf.sh`. You can refer to it for more details.
```bash
cd alignment
bash sample_train_safe_rlhf.sh
```

If you want to reproduce our results, we provide our first-iteration checkpoints in [this link](https://huggingface.co/zhezi12138/alpaca-7b-iter-1) and [this link](https://huggingface.co/zhezi12138/llama-3b-iter-1). You can download and train them. **Note:** We’ve retrained the models for more systematic results, and the results may slightly differ from that was reported in the paper. We will update it later.

**Evaluation**

For evaluation, we provide such a script, `sample_eval_safe_rlhf.sh` as well.
```bash
cd alignment
bash sample_eval_safe_rlhf.sh
```

We also provide a script, `sample_kl.sh`, for calculating the KL divergence between trained model and base model.
```bash
cd alignment
conda activate vllm
bash sample_kl.sh
```

**Clarification**

There are some details that we implement in a different way from RLHFlow. We use `rev_kl` instead of `kl` for `loss_type` during training, to align the setting closer to BT-model. We use the same set of prompts for each iteration, while `num_iter=0` refers to the test data. We also use different hyperparameters for generation, due to lack of computation resources.
