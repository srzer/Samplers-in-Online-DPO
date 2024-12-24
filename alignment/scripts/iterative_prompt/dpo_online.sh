num_iter=${1}
num_train_epochs=2
gradient_accumulation_steps=2
max_length=384
max_prompt_length=256
if [ "$num_iter" -gt 1 ]; then
  model_path="models/rlhflow_iter$((num_iter-1))"
else
  model_path=openlm-research/open_llama_3b_v2
fi
initial_model=openlm-research/open_llama_3b_v2
train_dir="./data/data_with_rewards_iter${num_iter}_online.json"
eval_dir="./data/data_with_rewards_iter${num_iter}_online.json"
accelerate launch --config_file ./configs/zero3.yaml ./dpo_iteration/run_dpo.py --run_name "rlhflow_iter${num_iter}" --output_dir "./models/rlhflow_iter${num_iter}" --model_name_or_path $model_path --ref_model $initial_model --train_dir $train_dir --eval_dir $eval_dir --learning_rate 5e-7 --max_steps 1200 --choose_type max_min --loss_type rev_kl --lr_scheduler_type cosine --max_length ${max_length} --max_prompt_length ${max_prompt_length} --gradient_checkpointing False --gradient_accumulation_steps $gradient_accumulation_steps --num_train_epochs $num_train_epochs --per_device_train_batch_size 2 --report_to none