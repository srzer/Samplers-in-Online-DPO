. "$HOME/miniconda3/etc/profile.d/conda.sh"

# conda activate vllm
# bash scripts/iterative_prompt/gen_online.sh 1 2
# conda activate rewardflow
# bash scripts/annotate.sh 1 online 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_online.sh 1

conda activate vllm
bash scripts/iterative_prompt/gen_mixp.sh 2 2
conda activate rewardflow
bash scripts/annotate_mixp.sh 2
conda activate rlhflow
bash scripts/iterative_prompt/dpo_mixp.sh 2

conda activate vllm
bash scripts/iterative_prompt/gen_mixp.sh 3 2
conda activate rewardflow
bash scripts/annotate_mixp.sh 3
conda activate rlhflow
bash scripts/iterative_prompt/dpo_mixp.sh 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_online.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 online 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_online.sh 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_online.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 online 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_online.sh 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_ref.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 ref 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_ref.sh 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_gshf.sh 2 2
# conda activate rewardflow
# bash scripts/annotate.sh 2 gshf 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_gshf.sh 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_ref.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 ref 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_ref.sh 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_gshf.sh 3 2
# conda activate rewardflow
# bash scripts/annotate.sh 3 gshf 2
# conda activate rlhflow
# bash scripts/iterative_prompt/dpo_gshf.sh 3