


def preprocess_function(examples, tokenizer, args.task_name, args.max_length, padding):
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

