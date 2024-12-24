num_iter=${1}
type=${2}
run_iter=${3}
seed=$num_iter
ref_model=openlm-research/open_llama_3b_v2
if [ $type == 'ref' ]; then
    infer_model=openlm-research/open_llama_3b_v2
elif [ $type == 'online' ]; then
    infer_model=models/rlhflow_iter$run_iter
else
    infer_model=models/rlhflow_iter${run_iter}_${type}
fi
my_world_size=4 # how many gpu you use
K=1 # how many samples you want to generate
prompt_dir=RLHFlow/iterative-prompt-v1-iter1-20K
output_dir=./data/gen_data

CUDA_VISIBLE_DEVICES=1 python ./generation/iterative_prompt/get_hf2.py --model_name_or_path ${infer_model} --ref_model_name_or_path ${ref_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature 0.1 --local_index 0 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 256 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=2 python ./generation/iterative_prompt/get_hf2.py --model_name_or_path ${infer_model} --ref_model_name_or_path ${ref_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature 0.1 --local_index 1 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 256 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=3 python ./generation/iterative_prompt/get_hf2.py --model_name_or_path ${infer_model} --ref_model_name_or_path ${ref_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature 0.1 --local_index 2 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 256 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} &
CUDA_VISIBLE_DEVICES=4 python ./generation/iterative_prompt/get_hf2.py --model_name_or_path ${infer_model} --ref_model_name_or_path ${ref_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature 0.1 --local_index 3 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length 256 --max_new_tokens 128 --num_iter ${num_iter} --seed ${seed} 
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/test_gen_data_iter${num_iter}_${type}_${run_iter}.json --num_datasets ${my_world_size}