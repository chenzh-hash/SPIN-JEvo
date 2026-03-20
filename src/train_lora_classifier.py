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
        learning_rate=5e-04,
        no_cuda=False,
        seed=8893,
        weight_decay=1e-03,
        num_train_epochs=3,
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_steps=500,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    peft_model = get_peft_model(model, peft_config)

    train_df = pd.read_csv(args.train_csv).sample(frac=1).reset_index(drop=True)
    train_df["Sequence"] = train_df["Sequence"].astype(str).str.slice(stop=1000)
    train_data = Dataset.from_pandas(train_df)

    def preprocess(example):
        tokenized = tokenizer(example["Sequence"])
        tokenized["label"] = example["label"]
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
