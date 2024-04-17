from datasets import load_dataset
import os
import numpy as np



def preprocess_function(examples, tokenizer, args, padding):
        if args.task_name == 'boolq':
            texts = [f"Answer the question with only True or False: {question} Context: {passage}" for passage, question in zip(examples['passage'], examples['question'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question_stem'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'ARC' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'winogrande' in  args.task_name:
            texts = [f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B {option2}. Answer:" for question, option1, option2 in zip(examples['sentence'], examples['option1'], examples['option2'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"1": 0, "2": 1, "":None}
            result["labels"] = [map_dict[label] for label in examples["answer"]]
        elif 'cola' in args.task_name:
            result = tokenizer(examples["sentence"], max_length=args.max_length, truncation=True, return_overflowing_tokens=False)
            result["labels"] = examples["label"]
        elif 'mnli' in args.task_name:
            result = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        elif 'sst2' in args.task_name:
            result = tokenizer(examples['sentence'], truncation=True, padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        elif 'stsb' in args.task_name:
            result = tokenizer(examples["sentence1"], examples["sentence2"],max_length = args.max_length,truncation=True,return_overflowing_tokens=False)
            result["labels"] = examples["label"]
        elif 'qnli' in args.task_name:
            result = tokenizer(examples["question"], examples["sentence"],max_length = args.max_length,truncation=True,return_overflowing_tokens=False)
            result["labels"] = examples["label"]
        elif 'qqp' in args.task_name:
            result = tokenizer(examples['question1'], examples['question2'], truncation=True, padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        elif 'rte'  in args.task_name:
            result = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        elif 'wnli' in args.task_name:
            result = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        elif 'mrpc' in args.task_name:
            result = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True,padding=padding, max_length=args.max_length)
            result["labels"] = examples["label"]
        return result

def convert_choices_to_alpha(example):
            # Define a mapping from numerical to alphabetical labels
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

            # Convert the 'label' field in 'choices'
            example['choices']['label'] = [mapping.get(label, label) for label in example['choices']['label']]

            # Convert the 'answerKey' field
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])

            example['choices']['text'] = [text if text.endswith('.') else text + '.' for text in example['choices']['text']]
            example['choices']['text'] = [text[0].upper() + text[1:] if text else text for text in example['choices']['text']]

            return example
        

def download_data(args,cache_dir):
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
            raw_datasets = load_dataset("glue", args.task_name)
            task_output_dir = os.path.join(cache_dir, f"metrics/glue/{args.task_name}/outputs/{args.task_name}/{args.model_name_or_path}")
            num_labels = len(np.unique(raw_datasets['train']['label']))
        elif args.task_name in ['cb', 'wic', 'boolq']:
            raw_datasets = load_dataset("super_glue", args.task_name) #/content/cache/huggingface/metrics/super_glue/boolq/outputs/boolq/google-bert
            task_output_dir = os.path.join(cache_dir, f"metrics/super_glue/{args.task_name}/outputs/{args.task_name}/{args.model_name_or_path}")
            num_labels = 2 #len(np.unique(raw_datasets['train']['answerKey']))
        elif 'ARC' in args.task_name:
            raw_datasets = load_dataset('ai2_arc', args.task_name)
            task_output_dir = os.path.join(cache_dir, f"metrics/accuracy/default/outputs/{args.task_name}/{args.model_name_or_path}")
            num_labels = 4 #len(np.unique(raw_datasets['train']['answerKey']))
        elif 'winogrande' in args.task_name:
            raw_datasets = load_dataset('winogrande', args.task_name)
            task_output_dir = os.path.join(cache_dir, f"metrics/accuracy/default/outputs/{args.task_name}/{args.model_name_or_path}")
            num_labels = len(np.unique(raw_datasets['train']['answer']))
        else:
            raw_datasets = load_dataset(args.task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        print(task_output_dir)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    if 'ARC' in args.task_name or 'openbookqa' in args.task_name:
        # Initialize counters
        count_3_choices_train = 0
        count_5_choices_train = 0
        count_3_choices_valid = 0
        count_5_choices_valid = 0

        # Count in the training dataset
        for example in raw_datasets["train"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_train += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_train += 1

        # Count in the validation dataset
        for example in raw_datasets["validation"]:
            if len(example['choices']['label']) == 3:
                count_3_choices_valid += 1
            elif len(example['choices']['label']) == 5:
                count_5_choices_valid += 1

        # Get total counts
        total_train = len(raw_datasets["train"])
        total_valid = len(raw_datasets["validation"])

        # Print counts
        print('====counts train====')
        print(f"Total number of training examples: {total_train}")
        print(f"Number of training questions with 3 choices: {count_3_choices_train}")
        print(f"Number of training questions with 5 choices: {count_5_choices_train}")

        print('====counts valid====')
        print(f"Total number of validation examples: {total_valid}")
        print(f"Number of validation questions with 3 choices: {count_3_choices_valid}")
        print(f"Number of validation questions with 5 choices: {count_5_choices_valid}")

        # Filter the examples in the training dataset
        filtered_train = raw_datasets["train"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the validation dataset
        filtered_valid = raw_datasets["validation"].filter(lambda example: len(example['choices']['label']) == 4)

        # Filter the examples in the test dataset
        filtered_test = raw_datasets["test"].filter(lambda example: len(example['choices']['label']) == 4)

        # Replace the original datasets with the filtered datasets
        raw_datasets["train"] = filtered_train
        raw_datasets["validation"] = filtered_valid
        raw_datasets["test"] = filtered_test

        print('====counts train====')
        print(f"Total number of training examples: {len(raw_datasets['train'])}")
        print('====counts valid====')
        print(f"Total number of validation examples: {len(raw_datasets['validation'])}")

        # Apply the conversion to the training, validation, and test datasets
        raw_datasets["train"] = raw_datasets["train"].map(convert_choices_to_alpha)
        raw_datasets["validation"] = raw_datasets["validation"].map(convert_choices_to_alpha)
        raw_datasets["test"] = raw_datasets["test"].map(convert_choices_to_alpha)

        print('====train data====')
        from collections import Counter

        # Initialize counters for training and validation datasets
        counter_train = Counter()
        counter_valid = Counter()

        # Count in the training dataset
        for example in raw_datasets["train"]:
            counter_train.update(example['answerKey'])

        # Count in the validation dataset
        for example in raw_datasets["validation"]:
            counter_valid.update(example['answerKey'])

        # Print the results
        print("Training dataset counts:")
        for choice, count in counter_train.items():
            print(f"Choice {choice}: {count} occurrences")

        print("Validation dataset counts:")
        for choice, count in counter_valid.items():
            print(f"Choice {choice}: {count} occurrences")
    return raw_datasets,num_labels



