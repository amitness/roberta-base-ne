import argparse
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)


def train_roberta_base(args):
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_architecture)
    config.save_pretrained(f"{output_dir}")

    dataset = load_dataset(
        "cc100",
        lang="ne",
        split=f"train[:{args.dataset_portion}]",
        trust_remote_code=True,
    )

    def batch_iterator(batch_size: int = 1000):
        for index in range(0, len(dataset), batch_size):
            yield dataset[index : index + batch_size]["text"]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=config.vocab_size,
        min_frequency=args.tokenizer_min_frequency,
        special_tokens=args.special_tokens,
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
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=args.fp16,
        dataloader_num_workers=cpu_count(),
        seed=args.seed,
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
    parser = argparse.ArgumentParser(description="Train RoBERTa base model for Nepali")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="roberta-base-ne",
        help="Output directory",
    )
    parser.add_argument(
        "--dataset-portion",
        type=str,
        default="100%",
        help="Portion of dataset to use",
    )
    parser.add_argument(
        "--tokenizer-min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokenizer",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        help="Special tokens for tokenizer",
    )
    parser.add_argument(
        "--model-architecture",
        type=str,
        default="roberta-base",
        help="Model architecture",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="MLM probability",
    )

    args = parser.parse_args()
    train_roberta_base(args)
