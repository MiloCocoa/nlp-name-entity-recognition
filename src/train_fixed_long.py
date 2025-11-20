import os
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback # <--- NEW IMPORT
)
from data_processing import NERDataset, LABEL_LIST, label2id, id2label
from metrics_utils import compute_metrics

# --- CONFIGURATION ---
MODEL_CHECKPOINT = "microsoft/deberta-v3-large" 
OUTPUT_DIR = "final_model_deberta_extended"

# Hyperparameters for Long Training
LR = 1e-5               # Reduced slightly (from 2e-5) for stability over long runs
BATCH_SIZE = 4          # Kept low for GPU memory safety
GRAD_ACCUMULATION = 4   # Effective Batch Size = 16
EPOCHS = 15             # Increased Ceiling (Early Stopping will likely cut this short)
MAX_LEN = 128

def main():
    print(f"Training {MODEL_CHECKPOINT} with Early Stopping...")
    
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    # 2. Load Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id
    )

    # 3. Load Fixed Splits
    # Check your folder structure! If data is in root, remove 'data/' prefix.
    # Assuming standard structure based on previous turns:
    print("Loading datasets...")
    train_path = "train_split.jsonl" if os.path.exists("train_split.jsonl") else "data/train_split.jsonl"
    val_path = "val_split.jsonl" if os.path.exists("val_split.jsonl") else "data/val_split.jsonl"
    
    train_dataset = NERDataset(train_path, tokenizer, max_len=MAX_LEN)
    val_dataset = NERDataset(val_path, tokenizer, max_len=MAX_LEN)

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",          # Evaluate every epoch
        save_strategy="epoch",          # Save checkpoint every epoch
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        
        # --- ANTI-OVERFITTING MEASURES ---
        weight_decay=0.05,              # Increased regularization (prevents weight explosion)
        warmup_ratio=0.1,               # 10% warmup to stabilize early training
        load_best_model_at_end=True,    # CRITICAL: Ensures we keep the best checkpoint, not the last
        metric_for_best_model="f1",     # Monitor F1 score
        greater_is_better=True,         # Higher F1 is better
        save_total_limit=2,             # Save disk space
        
        fp16=True,                      # Speed optimization
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 5. Initialize Trainer with Callback
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Add Early Stopping: Stop if F1 doesn't improve for 3 epochs
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # 6. Train
    trainer.train()
    
    # 7. Save Best Model
    print(f"Saving best model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()