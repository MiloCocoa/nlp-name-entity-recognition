import os
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from data_processing import NERDataset, LABEL_LIST, label2id, id2label
from metrics_utils import compute_metrics

# --- CONFIGURATION ---
# CHANGE THIS to "distilbert-base-uncased" for your second run!
# MODEL_CHECKPOINT = "microsoft/deberta-v3-large" 
MODEL_CHECKPOINT = "distilbert-base-uncased" 
OUTPUT_DIR = "final_model_distil_fixed" # Change to "final_model_distil" for second run

# Hyperparameters
LR = 2e-5
BATCH_SIZE = 16          # 4 for DeBERTa Large, 16 for DistilBERT
GRAD_ACCUMULATION = 1   # 4 for DeBERTa Large, 1 for DistilBERT
EPOCHS = 4
MAX_LEN = 128

def main():
    print(f"Training {MODEL_CHECKPOINT} on FIXED splits...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id
    )

    # LOAD FIXED FILES
    print("Loading datasets...")
    train_dataset = NERDataset("data/train_split.jsonl", tokenizer, max_len=MAX_LEN)
    val_dataset = NERDataset("data/val_split.jsonl", tokenizer, max_len=MAX_LEN)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()