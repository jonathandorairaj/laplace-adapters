# laplace-adapters

This repository contains code for the thesis titled : [Parameter-efficiency in Finetuning Pre-trained Large Language Models for Downstream
Tasks](https://liu.diva-portal.org/smash/get/diva2:1874935/FULLTEXT01.pdf)



## Acknowlegement 
This repository has been built from adamxyang/laplace-lora repository. Read more about Adam Yang et al.'s orginal paper at the following link : [Bayesian low-rank adaptation for large language models](https://arxiv.org/abs/2308.13111)

## Initialisation
This library is largely based on [Laplace](https://github.com/aleximmer/Laplace) and [ASDL](https://github.com/kazukiosawa/asdl/tree/master).

Before installing laplace-adapters, first install ASDL from source
```
pip install git+https://github.com/kazukiosawa/asdl
pip install accelerate peft datasets evaluate bitsandbytes -q
```

To install `laplace-adapters`, change directory to `laplace-adapters` and run 
```
pip install -e.
```

## LLM fine-tuning with LoRA
To fine-tune LlaMA2 or any GPT-like model on common sense reasoning tasks, use 
```
accelerate launch run_gpt.py
``` 
or the bash file 
```
bash run_gpt.sh
``` 
for submission to a slurm server. Customize training arguments like `lora_alpha`, `lora_r`, `lora_dropout`, etc. Set `testing_set` argument to `val` if using the full training set; set `testing_set` argument to `train_val` to split the training set into training and validation set.

### Hyperparameters for LoRA fine-tuning
There are several hyperparameters that can be tuned for LoRA fine-tuning, e.g. `lora_alpha`, `lora_r`, `lora_dropout`, `learning_rate`, etc.

To use the full training set and Laplace model evidence for optimizing Laplace prior precision, set  the `testing_set` argument to `val`; to split training set into a training set and a validation set and use minibatch gradient descent on the validation negative log-likelihood for optimizing Laplace prior precision, set the `testing_set` argument to `train_val`.

## Post-hoc Laplace-LoRA
To run post-hoc Laplace approximation on saved checkpoints, use 
``` 
accelerate launch run_gpt_laplace.py
``` 
or the bash file 
```
bash run_gpt_laplace.sh
``` 
for submission to a slurm server.

### Hyperparameters for Laplace-LoRA
To use full Laplace-LoRA, set the `laplace_sub` argument to `all`; to use last-layer Laplace-LoRA, set the `laplace_sub` argument to `last_layer`.

# Laplace-Adapters
Before we can fine-tuning LLMs using Adapters, first install the `adapters` library
```
pip install adapters
```

## LLM fine-tuning with Adapters
To fine-tune BERT or any BERT-like model on common sense reasoning tasks, use the following format
```
accelerate launch run_bert_adapters.py \
            --model_name_or_path "google-bert/bert-base-uncased" \
            --task_name 'mrpc' \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --learning_rate 1e-4 \
            --testing_set 'train_val' \
            --seed 65 \
            --max_length 256 \
            --lm_head \
            --max_length 256 \
            --num_train_epochs 5 \
            --reduction_factor 16 \
```

### Hyperparameters for Adapters fine-tuning
Several hyperparameters can be tuned for Adapters fine-tuning, however, in the current implementation, the `DoubleSeqConfig` is used with its' defaults. To add specific hyperparameters, simply refer the adapters library documentation and add the desired argument to the argument parser and pass the argument to `DoubleSeqConfig()`. Other adapters configurations can also be used by simple replacement. 

## Post-hoc Laplace Adapters 
To run post-hoc Laplace approximation on saved checkpoints, use 
```
accelerate launch run_laplace_adapters.py --laplace_sub "all" \
--model_name_or_path "roberta-base" \
--task_name "sst2" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--learning_rate 1e-4 \
--testing_set 'train_val' \
--seed 12345 \
--max_length 300 \
--use_slow_tokenizer \
--max_train_steps=10000 \
--checkpointing_steps 1000

```

### Hyperparameters for Laplace-Adapters
To change hyperparameters for the laplace approximations, please refer the arguments defined in ArugmentParser.

### Note

Please note the definition of a specified cache directory in the files. This was done to support running the scripts on Google Colab instances. Please change these directories to suit your machine.

The implementations here are for transformer-based models like BERT and RoBERTa. It is a trivial task to change the code to suit LLMs with other architechture. Be sure to modify the implementation details in LoRa and Adapters to match the architecture of the LLM chosen in your case.

