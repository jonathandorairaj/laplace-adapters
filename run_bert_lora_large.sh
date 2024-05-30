
#!/bin/bash

# List of GLUE tasks
tasks=("qqp" "mnli")

# Loop over each task and run the finetune_model.py script
for task in "${tasks[@]}"; do
  echo "Running finetuning for task: $task"
  accelerate launch run_gpt_bert.py   --model_name_or_path "bert-base-uncased"   --task_name "$task"   --per_device_train_batch_size=32   --per_device_eval_batch_size=32   --learning_rate=5e-5   --testing_set='train_val'   --seed=65   --max_length=300   --max_train_steps=10000   --checkpointing_steps 1000  
done
