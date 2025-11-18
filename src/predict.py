import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from data_processing import NERDataset, id2label, LABEL_LIST

# --- CONFIGURATION ---
MODEL_PATH = "final_model"        # Folder created by train.py
TEST_DATA_PATH = "data/test_data.jsonl" # Make sure this file is in your root or src folder
OUTPUT_FILE = "predict_output/submission.jsonl"
MAX_LEN = 128

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Model & Tokenizer
    try:
        # Try loading with fast tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        # Fallback for DeBERTa v3 if fast tokenizer fails
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    # 2. Load Test Data
    # We use is_test=True so the dataset doesn't look for 'ner_tags' columns
    print("Processing test data...")
    # Note: If test_data.jsonl is in the parent dir, use "../test_data.jsonl" or absolute path if needed.
    # Assuming it's in the same folder as where you run the script or defined relative to root.
    test_dataset = NERDataset(TEST_DATA_PATH, tokenizer, max_len=MAX_LEN, is_test=True)

    # 3. Initialize Trainer just for prediction
    trainer = Trainer(model=model)

    # 4. Run Inference
    print("Running prediction loop...")
    predictions, _, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    # 5. Decode and Realign
    print("Aligning predictions to original tokens...")
    
    submission_rows = []
    
    # Read raw data to ensure we match the exact token count required
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_rows = [json.loads(line) for line in f]

    for i, row in enumerate(raw_rows):
        pred_ids = predictions[i] # Shape: [128]
        original_tokens = row['tokens']
        
        # We need to re-tokenize to find which sub-words belong to which real word
        tokenized = tokenizer(
            original_tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors=None
        )
        word_ids = tokenized.word_ids()
        
        final_tags = []
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            # Skip special tokens
            if word_idx is None:
                continue
                
            # Take the tag of the FIRST sub-token of a word
            if word_idx != previous_word_idx:
                tag_id = pred_ids[idx]
                final_tags.append(id2label[tag_id])
                previous_word_idx = word_idx
        
        # SAFETY CHECK: Handle truncation or mismatched lengths
        if len(final_tags) < len(original_tokens):
            # If text was truncated, fill the rest with 'O'
            final_tags += ["O"] * (len(original_tokens) - len(final_tags))
        elif len(final_tags) > len(original_tokens):
             # Should rarely happen, but trim if necessary
            final_tags = final_tags[:len(original_tokens)]

        # Construct result row
        submission_rows.append({
            "id": row['id'],
            "tokens": row['tokens'],
            "ner_tags": final_tags
        })

    # 6. Save
    print(f"Saving {len(submission_rows)} rows to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for row in submission_rows:
            f.write(json.dumps(row) + "\n")
            
    print("Done! Prediction file generated.")

if __name__ == "__main__":
    main()