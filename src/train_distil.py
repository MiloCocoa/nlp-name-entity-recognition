import os
import torch
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from data_processing import NERDataset, LABEL_LIST, label2id, id2label
from metrics_utils import compute_metrics

# --- CONFIGURATION ---
MODEL_CHECKPOINT = "distilbert-base-uncased" # Distilled model
LR = 2e-5
BATCH_SIZE = 4          # Low batch size to fit on GPU
GRAD_ACCUMULATION = 4   # Accumulate gradients to simulate larger batch (4*4=16)
EPOCHS = 4
MAX_LEN = 128

def main():
    print(f"Loading model: {MODEL_CHECKPOINT}...")
    
    # 1. Load Tokenizer & Model
    # Note: use_fast=False fixes the DeBERTa conversion warning/error if it persists
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )

    # 2. Load Datasets
    print("Preparing datasets...")
    # We assume train_data.jsonl is in the PARENT directory based on your file structure
    # If it's inside src, change to "src/train_data.jsonl"
    data_path = "data/train_data.jsonl" 
    
    full_dataset = NERDataset(data_path, tokenizer, max_len=MAX_LEN)
    
    # Split Train/Validation (90/10 split)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3. Training Arguments
    args = TrainingArguments(
        output_dir="ner_model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,                     # Critical for RTX 3080 Ti speed
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",              # Disable wandb/mlflow for simplicity
        dataloader_num_workers=0       # Windows compatibility
    )

    # 4. Data Collator
    # Handles dynamic padding (pads to length of longest seq in batch, not max_len)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Final Evaluation & Save
    print("Evaluating best model...")
    trainer.evaluate()
    
    print("Saving model to 'final_model'...")
    trainer.save_model("final_model")

if __name__ == "__main__":
    main()