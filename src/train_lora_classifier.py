import argparse
import os

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, DataCollatorWithPadding, EsmForSequenceClassification, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA ESM-2 classifier on a labeled CSV.")
    parser.add_argument("--model-name", required=True, help="Base ESM-2 model name or local path.")
    parser.add_argument("--train-csv", required=True, help="Labeled CSV with Sequence and label columns.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained LoRA adapter.")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=8893)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.2,
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer_runs"),
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        report_to=[],
        remove_unused_columns=False,
    )

    peft_model = get_peft_model(model, peft_config)

    train_df = pd.read_csv(args.train_csv)
    missing = {args.sequence_column, args.label_column} - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_df[args.sequence_column] = (
        train_df[args.sequence_column].astype(str).str.slice(stop=args.max_length)
    )
    train_df[args.label_column] = train_df[args.label_column].astype(int)
    if not set(train_df[args.label_column]).issubset({0, 1}):
        raise ValueError("The label column must contain only binary labels 0 and 1")
    train_data = Dataset.from_pandas(train_df)

    def preprocess(example):
        tokenized = tokenizer(
            example[args.sequence_column],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["label"] = example[args.label_column]
        return tokenized

    train_data = train_data.map(preprocess, remove_columns=train_data.column_names, batched=True)
    print(train_data[0:2])

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    tokenizer.save_pretrained(args.output_dir)
    peft_model.save_pretrained(args.output_dir)

    print(f"Saved model to: {args.output_dir}")
    print(f"Training samples: {len(train_data)}")


if __name__ == "__main__":
    main()
