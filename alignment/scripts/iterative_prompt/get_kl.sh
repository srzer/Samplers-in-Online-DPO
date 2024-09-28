num_iter=2
my_world_size=1 # how many gpu you use
K=1 # how many samples you want to generate
max_input_length=256
max_new_tokens=128
seed=$num_iter
ref_model=openlm-research/open_llama_3b_v2
infer_model=${1}
prompt_dir=RLHFlow/iterative-prompt-v1-iter1-20K
output_dir=./data/gen_data
temperature=0.7

CUDA_VISIBLE_DEVICES=5,6 python ./generation/iterative_prompt/get_kl.py --model_name_or_path ${infer_model} --ref_model_name_or_path ${ref_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K ${K} --temperature ${temperature} --local_index 0 --my_world_size ${my_world_size} --eos_ids 2 --max_input_length ${max_input_length} --max_new_tokens ${max_new_tokens} --num_iter ${num_iter} --seed ${seed} 