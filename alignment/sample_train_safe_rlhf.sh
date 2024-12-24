. "$HOME/miniconda3/etc/profile.d/conda.sh"

# conda activate vllm
# bash scripts/safe_rlhf/gen_online.sh 1 2
# conda activate rewardflow
# bash scripts/annotate.sh 1 online 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_online.sh 1

# conda activate vllm
# bash scripts/safe_rlhf/gen_online.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 online 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_online.sh 2

# conda activate vllm
# bash scripts/safe_rlhf/gen_online.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 online 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_online.sh 3

conda activate vllm
bash scripts/safe_rlhf/gen_mixp.sh 2 2
conda activate rewardflow
bash scripts/annotate.sh 2 mixp 2
conda activate rlhflow
bash scripts/safe_rlhf/dpo_mixp.sh 2

conda activate vllm
bash scripts/safe_rlhf/gen_mixp.sh 3 2
conda activate rewardflow
bash scripts/annotate.sh 3 mixp 2
conda activate rlhflow
bash scripts/safe_rlhf/dpo_mixp.sh 3

# conda activate vllm
# bash scripts/safe_rlhf/gen_ref.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 ref 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_ref.sh 2

# conda activate vllm
# bash scripts/safe_rlhf/gen_gshf.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 gshf 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_gshf.sh 2

# conda activate vllm
# bash scripts/safe_rlhf/gen_ref.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 ref 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_ref.sh 3

# conda activate vllm
# bash scripts/safe_rlhf/gen_gshf.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 gshf 2
# conda activate rlhflow
# bash scripts/safe_rlhf/dpo_gshf.sh 3