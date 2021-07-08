import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)


@hydra.main(config_path="config", config_name="default.yaml")
def train_roberta_base(cfg: DictConfig):
    output_dir = cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(cfg.model.architecture)
    config.save_pretrained(f"{output_dir}")

    dataset = load_dataset("cc100", lang="ne", split=f"train[:{cfg.dataset.portion}]")

    def batch_iterator(batch_size: int = 1000):
        for index in range(0, len(dataset), batch_size):
            yield dataset[index : index + batch_size]["text"]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=config.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=list(cfg.tokenizer.special_tokens),
    )
    tokenizer.save(f"{output_dir}/tokenizer.json")

    tokenizer = RobertaTokenizerFast.from_pretrained(output_dir, max_len=512)
    model = AutoModelForMaskedLM.from_config(config)

    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding=True),
        batched=True,
        num_proc=cpu_count(),
        remove_columns=["text"],
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.model.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        overwrite_output_dir=True,
        num_train_epochs=cfg.model.epochs,
        per_device_train_batch_size=cfg.model.batch_size,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=cfg.model.fp16,
        dataloader_num_workers=cpu_count(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    train_roberta_base()
