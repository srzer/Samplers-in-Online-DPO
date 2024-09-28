num_iter=${1}
K=2
CUDA_VISIBLE_DEVICES=1 python ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data_iter${num_iter}_mixp_temp1.json --output_dir ./data/data_with_rewards_iter${num_iter}_mixp_temp1.json --K ${K} --reward_name_or_path 'sfairXC/FsfairX-LLaMA3-RM-v0.1' &
CUDA_VISIBLE_DEVICES=2 python ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data_iter${num_iter}_mixp_temp2.json --output_dir ./data/data_with_rewards_iter${num_iter}_mixp_temp2.json --K ${K} --reward_name_or_path 'sfairXC/FsfairX-LLaMA3-RM-v0.1'
wait 
python ./generation/concat.py --num_iter ${num_iter}