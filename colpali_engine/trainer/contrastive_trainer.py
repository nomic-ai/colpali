import time
import torch
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.utils.data import BatchSampler, Dataset, ConcatDataset, DataLoader
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, List, Optional
from datasets import DatasetDict

class SingleDatasetBatchSampler(BatchSampler):
    """
    A batch sampler that samples from a single dataset per batch and handles distribution across GPUs.
    
    Args:
        datasets (List[Dataset]): List of datasets to sample from
        batch_size (int): Global batch size (will be divided across GPUs)
        drop_last (bool): Whether to drop the last incomplete batch
        generator (Optional[torch.Generator]): Random number generator
    """
    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        
        # Calculate dataset sizes and create index mappings
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        #### get start of each dataset #####
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)
        
        # Create shuffled indices for each dataset
        self.indices_per_dataset = [
            torch.randperm(size, generator=self.generator).tolist()
            for size in self.dataset_sizes
        ]
        self.current_positions = [0] * len(datasets)

    def __iter__(self) -> Iterator[List[int]]:
        while True:
            # Randomly select a dataset
            dataset_idx = torch.randint(len(self.datasets), size=(1,), generator=self.generator).item()
            
            # Get indices for the current dataset
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]
            
            # Get batch indices
            batch_indices = [
                idx + self.cumsum_sizes[dataset_idx]
                for idx in dataset_indices[current_pos:current_pos + self.global_batch_size]
            ]
            # Update position
            self.current_positions[dataset_idx] = current_pos + self.global_batch_size
            yield batch_indices

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        if self.drop_last:
            return sum(size // self.global_batch_size for size in self.dataset_sizes)
        else:
            return sum((size + self.global_batch_size - 1) // self.global_batch_size for size in self.dataset_sizes)


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
            # round down each dataset if not divible by global batch size
            batch_size = kwargs["args"].train_batch_size
            for i in range(len(dataset_list)):
                if len(dataset_list[i]) % batch_size != 0:
                    total_samples = (len(dataset_list[i]) // batch_size) * batch_size
                    dataset_list[i] = dataset_list[i].take(total_samples)
            
            kwargs["train_dataset"] = ConcatDataset(dataset_list)
        else:
            dataset_list = None

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.dataset_list = dataset_list

        
    def get_train_dataloader(self):
        ######## adapted from trainer (gross) ########
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            ###### batch_sampler set instead of sampler in trainer code #######
            dataloader_params["batch_sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        
    def _get_train_sampler(self):
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None
