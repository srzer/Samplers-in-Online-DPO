num_iter=${1}
my_world_size=4 # how many gpu you use
K=${2} # how many samples you want to generate
seed=$num_iter
infer_model='PKU-Alignment/alpaca-7b-reproduced'
prompt_dir=PKU-Alignment/PKU-SafeRLHF-prompt
output_dir=./data/gen_data
temperature=0.7

CUDA_VISIBLE_DEVICES=0 python ./generation/safe_rlhf/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature ${temperature} --local_index 0 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 128 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=1 python ./generation/safe_rlhf/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature ${temperature} --local_index 1 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 128 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=2 python ./generation/safe_rlhf/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature ${temperature} --local_index 2 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 128 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=3 python ./generation/safe_rlhf/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature ${temperature} --local_index 3 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 128 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} 

wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/gen_data_iter${num_iter}_ref.json --num_datasets ${my_world_size}