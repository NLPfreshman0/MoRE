from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, logging
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, Tuple, Union, List
import torch.nn.functional as F
import torch
from data.multi_task_sample import MultiTaskDataLoader, TaskCollator
import math
import numpy as np
from rank import *

logger = logging.get_logger(__name__)

class MyTrainer(Trainer):
    def __init__(self, config=None, tokenizer=None, data_args=None,  *args, **kwargs):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.config = config
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        
    def get_train_dataloader(self):
        return MultiTaskDataLoader(
            datasets=self.train_dataset,
            data_collator=self.data_collator,
            batch_size=self.args.train_batch_size,
            data_args=self.data_args,
            seed=self.args.seed
        )
        
    # def get_eval_dataloader(self, eval_dataset):
    #     return DataLoader(
    #         dataset=eval_dataset,
    #         collate_fn=self.data_collator,
    #         batch_size=self.args.eval_batch_size,
    #         drop_last=False
    #     )
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": 10,
            "num_beams": self.config.num_beams
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            # compute loss on predict data
            loss = self.compute_loss(model, inputs)
        
        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens
        
        labels = inputs.pop("labels")
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id`"
                f" is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for index, (eval_dataset_name, eval_dataset) in enumerate(self.eval_dataset.items()):
                    set_task_id(index)
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
                metric = [metrics[key] for key in metrics.keys() if "acc" in key or 'mcc' in key or 'pearson' in key]
                metrics['eval_average_metrics'] = np.mean(metric)
                losses = [metrics[key] for key in metrics.keys() if "loss" in key]
                metrics['eval_loss'] = np.mean(losses)
                print({'eval_average_metrics': metrics['eval_average_metrics'], 'eval_average_loss': metrics['eval_loss']})
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
