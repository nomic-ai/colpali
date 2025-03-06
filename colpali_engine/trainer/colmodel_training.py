import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.trainer.eval_utils import BenchmarkEvalCallback, CustomRetrievalEvaluator
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: PreTrainedModel
    tr_args: TrainingArguments = None
    output_dir: str = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    processor: BaseVisualRetrieverProcessor = None
    tokenizer: PreTrainedTokenizer = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None:
            self.tr_args.output_dir = self.output_dir

        # cast if string
        if isinstance(self.tr_args.learning_rate, str):
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)
        self.tr_args.remove_unused_columns = False

        if self.processor is None and self.tokenizer is None:
            print("Using textual model tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)

        if self.pretrained_peft_model_name_or_path is not None:
            self.model.load_adapter(self.pretrained_peft_model_name_or_path)

            print(f"Loaded pretrained adapter from {self.pretrained_peft_model_name_or_path}")

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.processor is None:
                # Might be deprecated - use the "else" branch
                self.model = prepare_model_for_kbit_training(self.model)  # use_gradient_checkpointing=True
                # self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                if self.pretrained_peft_model_name_or_path is None:
                    # self.model.add_adapter(self.peft_config)
                    # self.model.enable_adapters()
                    self.model = get_peft_model(self.model, self.peft_config)
                    self.model.print_trainable_parameters()
                else:
                    print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.dataset = self.config.dataset_loading_func()
        if isinstance(self.dataset, Tuple):
            corpus_format = self.dataset[2]
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = CorpusQueryCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                image_dataset=neg_dataset,
                mined_negatives=True,
                corpus_format=corpus_format,
            )
        else:
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
            )
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.retrieval_evaluator = CustomRetrievalEvaluator()

    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )

        trainer.args.remove_unused_columns = False

        if self.config.processor is not None:
            trainer.add_callback(
                BenchmarkEvalCallback(
                    processor=self.config.processor,
                    model=self.model,
                    eval_dataset_loader=self.config.eval_dataset_loader,
                    batch_query=self.config.tr_args.per_device_eval_batch_size,
                    batch_passage=4,
                    batch_score=4,
                    run_frequency=getattr(self.config.tr_args, "eval_steps_frequency", 500),
                    dataset_format=getattr(self.config.tr_args, "eval_dataset_format", "beir"),
                )
            )

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def save(self, config_file):
        # save model
        self.model.save_pretrained(self.config.output_dir)
        if self.config.tokenizer is not None:
            self.config.tokenizer.save_pretrained(self.config.output_dir)
        if self.config.processor is not None:
            self.config.processor.save_pretrained(self.config.output_dir)  # save config

        # copy-paste the yml file with os
        os.system(f"cp {config_file} {self.config.output_dir}/training_config.yml")

        # save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
