import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Specify which GPU to use
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)
from transformers import (
    AutoModelForTokenClassification,  # Model for token classification tasks
    AutoTokenizer,  # Tokenizer to preprocess text inputs
    DataCollatorForTokenClassification,  # Data collator to batch token sequences
    TrainingArguments,  # Arguments for setting training configurations
    Trainer,  # HuggingFace Trainer class for handling the training loop
    AutoModel  # Base model class for transformers
)
from datasets import Dataset
from accelerate import Accelerator  # Used for accelerating training across multiple devices

# PEFT (Parameter-Efficient Fine-Tuning) specific imports for LoRA fine-tuning
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import AutoTokenizer, EsmForSequenceClassification
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Tokenizer and Pretrained Model (ESM2, a protein transformer)
tokenizer = AutoTokenizer.from_pretrained("/mnt/hdd/chenzh/esm2_t33_650M_UR50D")
model = EsmForSequenceClassification.from_pretrained("/mnt/hdd/chenzh/esm2_t33_650M_UR50D", num_labels=2)
t=80
# Configuration for LoRA fine-tuning with specific parameters
config = {
    "lora_alpha": 16,  # Scaling factor for LoRA fine-tuning
    "lora_dropout": 0.2,  # Dropout rate to avoid overfitting
    "lr": 5e-04,  # Learning rate for the optimizer
    "lr_scheduler_type": "cosine",  # Scheduler to adjust learning rate during training
    "max_grad_norm": 0.5,  # Maximum gradient norm for gradient clipping
    "num_train_epochs": 1,  # Number of epochs for training
    "per_device_train_batch_size": 12,  # Batch size for each device during training
    "r": 16,  # Rank of the LoRA matrix (controls the bottleneck size)
    "weight_decay": 0.001  # Weight decay for preventing overfitting
}

# Set up the LoRA configuration for fine-tuning specific layers in the transformer model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    inference_mode=False,  # Indicates training mode (not inference)
    r=config["r"],  # LoRA rank parameter
    lora_alpha=config["lora_alpha"],  # LoRA scaling parameter
    target_modules=["query", "key", "value"],  # Attention layers to apply LoRA
    lora_dropout=config["lora_dropout"],  # Dropout during fine-tuning
    bias="none"  # Bias setup, can be "none", "all", or "lora_only"
)

# Training arguments for controlling how the model is trained (batch size, learning rate, etc.)
training_args = TrainingArguments(
    output_dir='./ccda_ecoli',  # Where to save the model checkpoints
    #evaluation_strategy='steps',  # Evaluate every few steps
    learning_rate=config['lr'],  # Learning rate
    no_cuda=False,  # Whether or not to use GPU
    seed=8893,  # Random seed for reproducibility
    weight_decay=1e-03,  # Weight decay
    num_train_epochs=8,  # Number of epochs
    save_strategy='epoch',  # Save the model at the end of each epoch
    per_device_train_batch_size=10,  # Batch size per device
    per_device_eval_batch_size=4,  # Evaluation batch size
    eval_steps=500,  # Evaluate every 500 steps
    report_to='tensorboard',
    remove_unused_columns=False# Report results to Tensorboard for visualization
)
peft_model = get_peft_model(model, peft_config)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Load the training data
X_train = pd.read_csv('Antitoxin_CcdA_complex.csv').sample(frac=1).reset_index(drop=True)
X_train['Sequence'] = X_train['Sequence'].str.slice(stop=1000)  # Limit sequences to 1000 characters

# Load the test data
X_test = pd.read_csv('Antitoxin_CcdA_complex.csv').sample(frac=1).reset_index(drop=True)
X_test['Sequence'] = X_test['Sequence'].str.slice(stop=1000)

# Convert the pandas DataFrame to HuggingFace Dataset format for processing
data_train = Dataset.from_pandas(X_train)
data_valid = Dataset.from_pandas(X_test)

# Preprocessing function to tokenize the sequences and attach labels
def preprocess(example):
    tokenized_example = tokenizer(example["Sequence"])  # Tokenize the protein sequence
    tokenized_example['label'] = example['label']  # Attach the label
    return tokenized_example

# Apply preprocessing to both the training and validation datasets
data_train = data_train.map(preprocess, remove_columns=data_train.column_names, batched=True)
data_valid = data_valid.map(preprocess, remove_columns=data_valid.column_names, batched=True)
print(data_train[0:2])
# Function to create the Trainer object that handles model training
def get_trainer():
    return Trainer(
        model=peft_model,  # The LoRA fine-tuned model
        args=training_args,  # Training arguments
        train_dataset=data_train,  # Training dataset
        eval_dataset=data_valid,  # Validation dataset
        tokenizer=tokenizer,  # Tokenizer for preprocessing
        compute_metrics=compute_metrics,  # Function to compute accuracy and other metrics
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)  # Collator to pad sequences to the same length
    )
def train_save():
    # Train the model using the Trainer class
    peft_lora_finetuning_trainer = get_trainer()

    peft_lora_finetuning_trainer.train()
    #peft_lora_finetuning_trainer.evaluate()

    # Save the tokenizer and the fine-tuned model for future use
    tokenizer.save_pretrained('ccda0_complex_model')
    peft_model.save_pretrained('ccda0_complex_model')
train_save()
print('data_train:',len(data_train))
print('data_vali:',len(data_valid))
