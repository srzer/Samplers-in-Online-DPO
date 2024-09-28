num_iter=${1}
type=${2}
K=${3}
CUDA_VISIBLE_DEVICES=1 python ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data_iter${num_iter}_${type}.json --output_dir ./data/data_with_rewards_iter${num_iter}_${type}.json --K ${K} --reward_name_or_path 'sfairXC/FsfairX-LLaMA3-RM-v0.1'