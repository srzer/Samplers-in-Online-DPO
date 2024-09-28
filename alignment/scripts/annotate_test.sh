num_iter=${1}
type=${2}
run_iter=${3}
CUDA_VISIBLE_DEVICES=1 python ./annotate_data/get_rewards.py --dataset_name_or_path ./data/test_gen_data_iter${num_iter}_${type}_${run_iter}.json --output_dir ./data/test_data_with_rewards_iter${num_iter}_${type}_${run_iter}.json --K 1 --reward_name_or_path 'sfairXC/FsfairX-LLaMA3-RM-v0.1'