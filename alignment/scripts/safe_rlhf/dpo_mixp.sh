num_iter=${1}
num_train_epochs=2
gradient_accumulation_steps=2
max_length=256
max_prompt_length=128
if [ "$num_iter" -gt 2 ]; then
  model_path="models/rlhflow_iter$((num_iter-1))_mixp"
else
  model_path="models/rlhflow_iter1"
fi
initial_model='PKU-Alignment/alpaca-7b-reproduced'
train_dir="./data/data_with_rewards_iter${num_iter}_mixp.json"
eval_dir="./data/data_with_rewards_iter${num_iter}_mixp.json"
accelerate launch --config_file ./configs/zero3.yaml ./dpo_iteration/run_dpo.py --run_name "rlhflow_iter${num_iter}_mixp" --output_dir "./models/rlhflow_iter${num_iter}_mixp" --model_name_or_path $model_path --ref_model $initial_model --train_dir $train_dir --eval_dir $eval_dir --learning_rate 5e-7 --max_steps 1200 --choose_type max_min --loss_type rev_kl --lr_scheduler_type cosine --max_length ${max_length} --max_prompt_length ${max_prompt_length} --gradient_checkpointing False --gradient_accumulation_steps $gradient_accumulation_steps --num_train_epochs $num_train_epochs --margin_scale 4 --per_device_train_batch_size 1  --report_to none