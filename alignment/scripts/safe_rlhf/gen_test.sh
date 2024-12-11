num_iter=${1}
type=${2}
run_iter=${3}
if [ $type == 'ref' ]; then
    infer_model='PKU-Alignment/alpaca-7b-reproduced'
elif [ $type == 'online' ]; then
    infer_model=models/rlhflow_iter$run_iter
else
    infer_model=models/rlhflow_iter${run_iter}_$type
fi
my_world_size=1 # how many gpu you use
K=1 # how many samples you want to generate
prompt_dir=PKU-Alignment/PKU-SafeRLHF-prompt
output_dir=./data/gen_data

CUDA_VISIBLE_DEVICES=7 python ./generation/safe_rlhf/get_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature 0.1 --local_index 0 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 128 --max_new_tokens 128 --num_iter ${num_iter}

wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/test_gen_data_iter${num_iter}_${type}_${run_iter}.json --num_datasets ${my_world_size}