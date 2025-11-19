import json
import random
import os

# Config
INPUT_FILE = "data/train_data.jsonl"
TRAIN_OUTPUT = "data/train_split.jsonl"
VAL_OUTPUT = "data/val_split.jsonl"
SPLIT_RATIO = 0.9
SEED = 42

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Deterministic Shuffle
    random.seed(SEED)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * SPLIT_RATIO)
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]
    
    print(f"Writing {len(train_data)} samples to {TRAIN_OUTPUT}...")
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
        
    print(f"Writing {len(val_data)} samples to {VAL_OUTPUT}...")
    with open(VAL_OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
        
    print("Done. Physical splits created.")

if __name__ == "__main__":
    main()