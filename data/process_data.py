from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
from datasets import Dataset
import json
import numpy as np


dataset_path = "glue"
glue_tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']
prompt_templates = {
    'cola': "cola sentence: {sentence}",
    'mnli': "mnli hypothesis: {hypothesis} premise: {premise}",
    'mrpc': "mrpc sentence1: {sentence1} sentence2: {sentence2}",
    'qnli': "qnli question: {question} sentence: {sentence}",
    'qqp': "qqp question1: {question1} question2: {question2}",
    'rte': "rte sentence1: {sentence1} sentence2: {sentence2}",
    'sst2': "sst2 sentence: {sentence}",
    'stsb': "stsb sentence1: {sentence1} sentence2: {sentence2}"
}
def label_map(label, task_name):
    if task_name == 'cola':
        label = {0: 'unacceptable', 1: 'acceptable'}.get(int(label), 'unknown label')
    elif task_name ==  'mnli':
        label = {0: 'entailment', 1: 'neutral', 2:'contradiction'}.get(int(label), 'unknown label')
    elif task_name == 'mrpc':
        label = {0: 'not_equivalent', 1: 'equivalent'}.get(int(label), 'unknown label')
    elif task_name == 'qnli':
        label = {0: 'entailment', 1: 'not_entailment'}.get(int(label), 'unknown label')
    elif task_name == 'qqp':
        label = {0: 'not_duplicate', 1: 'duplicate'}.get(int(label), 'unknown label')
    elif task_name == 'rte':
        label = {0: 'entailment', 1: 'not_entailment'}.get(int(label), 'unknown label')
    elif task_name == 'sst2':
        label = {0: 'negative', 1: 'positive'}.get(int(label), 'unknown label')
    elif task_name == 'stsb':
        label = "{:.1f}".format(round(float(label) * 5) / 5)
    else:
        print('task_name error')
    if label == 'unknown label' and task_name != 'stsb':
        print('label error')
        return '-1'
    else:
        return label

def add_prompt(dataset, task_name):
    columns = dataset.column_names
    prompts = []
    labels = []
    template = prompt_templates.get(task_name, "Task not found in the dictionary.")
    for sample in tqdm(dataset):
        prompt = template.format(**sample)
        prompts.append(prompt)
        label = label_map(sample['label'], task_name)
        labels.append(label)
        #print(prompt, label)
    dataset = dataset.add_column("prompt", prompts)
    dataset = dataset.remove_columns(columns)  
    dataset = dataset.add_column("label", labels)
    return dataset

def remove_error_label(dataset, task_name):
    filtered_dataset = dataset.filter(lambda example: example['label'] != '-1')
    return filtered_dataset


for task_name in glue_tasks:
    print(task_name)
    task_load_path = f"{dataset_path}/{task_name}"
    dataset = load_from_disk(task_load_path)
    print(dataset)
    splits = []
    for split in dataset:
        if 'test' in split:
            continue
        splits.append(split)
        data = dataset[split]
        print(f'process {split}')
        data = add_prompt(data, task_name)
        if 'test' not in split:
            data = remove_error_label(data, task_name)
        data.save_to_disk(f'{dataset_path}/{task_name}_with_prompt/{split}')
    dataset_dict = {
        "splits": splits
    }
    json_path = f'{dataset_path}/{task_name}_with_prompt/dataset_dict.json'
    with open(json_path, "w") as json_file:
        json.dump(dataset_dict, json_file)
