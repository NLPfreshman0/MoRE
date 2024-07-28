from data.multi_task_sample import AutoTask, TaskCollator, MultiTaskDataLoader
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, HfArgumentParser, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional, List, Union
from dataclasses import dataclass, field
from trainer import MyTrainer
import torch
import random
import numpy as np
import os
import datasets
from rank import *
from metrics import accuracy, pearson_corrcoef, matthews_corrcoef

@dataclass
class MyArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tasks: Optional[List[str]] = field(default_factory=lambda: ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2'])
    max_length: Optional[int] = field(default=128)
    use_lora: bool = field(default=True)
    lora_rank: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["query", "key", "value", "decoder"])
    epochs: Optional[int] = field(default=10)
    smooth_distribution: bool = field(default=True)
    sample_by_loss: bool = field(default=False)
    use_dyrank: bool = field(default=False)
    use_share_module: bool = field(default=False)

parser = HfArgumentParser((TrainingArguments, MyArguments))
training_args, data_args = parser.parse_args_into_dataclasses()
training_args.remove_unused_columns = False

torch.manual_seed(training_args.seed)
torch.cuda.manual_seed_all(training_args.seed)
np.random.seed(training_args.seed)
random.seed(training_args.seed)

#tasks = ['stsb']
tasks = data_args.tasks
print(tasks)
#['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
dataset_class = AutoTask
train_datasets = [dataset_class.get(task).get_dataset(
    split="train") for task in tasks]#1238 1079
eval_datasets = ({task: dataset_class.get(task, seed=1189).get_dataset(
                split="validation") for task in tasks})

dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
print(train_datasets)
print({tasks[i]:dataset_sizes[i] for i in range(len(tasks))})
# train_datasets = datasets.concatenate_datasets(train_datasets)
# print(train_datasets)

# 加载 RoBERTa tokenizer
config = T5Config.from_pretrained(data_args.model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(data_args.model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(data_args.model_name_or_path)

if data_args.use_lora:
    lora_config = LoraConfig(
        r=data_args.lora_rank,
        lora_alpha=data_args.lora_alpha,
        target_modules=data_args.target_modules,
        # 即论文图中右边两个模块输出的dropout
        lora_dropout=0.1,
        bias="none",
        # 说明任务类型，从而影响模型的架构，loss，输出形式
        #task_type='CAUSAL_LM'
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    print(data_args.target_modules)
    model.print_trainable_parameters() 
    #print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
def lmap(f, x) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def compute_metrics(eval_prediction):
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    #print(predictions, label_ids)
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_ids[label_ids == -100] = 0
    #print(predictions, label_ids)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #print(pred_str, label_str)
    pred_str = lmap(str.strip, pred_str)
    label_str = lmap(str.strip, label_str)

    if get_task_id() == 0: 
        #print(pred_str)hhbt
        acc = matthews_corrcoef(pred_str, label_str)
    elif get_task_id() == 7:
        pred_str = [float(pred) if pred.replace('.', '', 1).isdigit() else 0.0 for pred in pred_str]
        label_str = [float(label) for label in label_str]
        acc = pearson_corrcoef(pred_str, label_str)
    else:
        acc = accuracy(pred_str, label_str)
    return acc

        
my_trainer = MyTrainer(model=model, 
                       config=config,
                        args=training_args,
                        train_dataset=train_datasets,
                        eval_dataset=eval_datasets,
                        data_collator=TaskCollator(tokenizer, data_args=data_args),
                        #data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        )


my_trainer.train()