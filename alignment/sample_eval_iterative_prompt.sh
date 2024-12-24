. "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate vllm
bash scripts/iterative_prompt/gen_test.sh 2 ref 2
bash scripts/iterative_prompt/gen_test.sh 0 ref 2
conda activate rewardflow
bash scripts/annotate_test.sh 2 ref 2
bash scripts/annotate_test.sh 0 ref 2

conda activate vllm
bash scripts/iterative_prompt/gen_test.sh 2 mixp 2
bash scripts/iterative_prompt/gen_test.sh 0 mixp 2
conda activate rewardflow
bash scripts/annotate_test.sh 2 mixp 2
bash scripts/annotate_test.sh 0 mixp 2

conda activate vllm
bash scripts/iterative_prompt/gen_test.sh 3 mixp 3
bash scripts/iterative_prompt/gen_test.sh 0 mixp 3
conda activate rewardflow
bash scripts/annotate_test.sh 3 mixp 3
bash scripts/annotate_test.sh 0 mixp 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 2 online 2
# bash scripts/iterative_prompt/gen_test.sh 0 online 2
# conda activate rewardflow
# bash scripts/annotate_test.sh 2 online 2
# bash scripts/annotate_test.sh 0 online 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 2 offline 2
# bash scripts/iterative_prompt/gen_test.sh 0 offline 2
# conda activate rewardflow
# bash scripts/annotate_test.sh 2 offline 2
# bash scripts/annotate_test.sh 0 offline 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 2 gshf 2
# bash scripts/iterative_prompt/gen_test.sh 0 gshf 2
# conda activate rewardflow
# bash scripts/annotate_test.sh 2 gshf 2
# bash scripts/annotate_test.sh 0 gshf 2

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 3 online 3
# bash scripts/iterative_prompt/gen_test.sh 0 online 3
# conda activate rewardflow
# bash scripts/annotate_test.sh 3 online 3
# bash scripts/annotate_test.sh 0 online 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 3 offline 3
# bash scripts/iterative_prompt/gen_test.sh 0 offline 3
# conda activate rewardflow
# bash scripts/annotate_test.sh 3 offline 3
# bash scripts/annotate_test.sh 0 offline 3

# conda activate vllm
# bash scripts/iterative_prompt/gen_test.sh 3 gshf 3
# bash scripts/iterative_prompt/gen_test.sh 0 gshf 3
# conda activate rewardflow
# bash scripts/annotate_test.sh 3 gshf 3
# bash scripts/annotate_test.sh 0 gshf 3