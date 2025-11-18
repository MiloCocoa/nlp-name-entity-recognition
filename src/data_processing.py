import json
import spacy
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

# --- CONFIGURATION ---
# Exact labels from PDF 
LABEL_LIST = [
    "O",
    "B-Politician", "I-Politician",
    "B-Artist", "I-Artist",
    "B-Facility", "I-Facility",
    "B-HumanSettlement", "I-HumanSettlement",
    "B-OtherPER", "I-OtherPER",
    "B-PublicCorp", "I-PublicCorp",
    "B-ORG", "I-ORG"
]

# Create Mappings
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}

# Load Spacy for POS Tagging (External Feature Requirement)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, is_test=False):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
        # Load and process data
        self.load_data(data_path)

    def load_data(self, path):
        """
        Loads JSONL and adds POS tags as external features.
        """
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                tokens = row['tokens']
                
                # FEATURE ENGINEERING: POS Tags [cite: 66, 82]
                # We use spacy to get POS tags for the existing tokens
                # Note: We construct a Doc from the pre-tokenized list to match 1:1
                doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
                nlp(doc) # Run pipeline
                pos_tags = [token.pos_ for token in doc] # e.g., ['PROPN', 'VERB']
                
                entry = {
                    'id': row['id'],
                    'tokens': tokens,
                    'pos_tags': pos_tags, 
                    # Prepare labels if training data
                    'ner_tags': row.get('ner_tags', [])
                }
                self.data.append(entry)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        ner_labels = item['ner_tags'] # List of strings "B-Artist", etc.
        
        # 1. Tokenize and align labels
        # The tokenizer splits "Chatuchak" -> ["Cha", "##tu", "##chak"]
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None # Return python lists
        )
        
        word_ids = tokenized_inputs.word_ids() # Maps tokens to original word index
        
        label_ids = []
        previous_word_idx = None
        
        # 2. Label Alignment Logic
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens [CLS], [SEP], [PAD] get ignored
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word -> Assign the actual label
                if not self.is_test:
                    label_str = ner_labels[word_idx]
                    label_ids.append(label2id[label_str])
                else:
                    label_ids.append(-100) # Dummy for test
            else:
                # Subsequent subwords -> Ignore (-100)
                label_ids.append(-100)
            previous_word_idx = word_idx

        # Return format compatible with HF Trainer
        return {
            'input_ids': torch.tensor(tokenized_inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            # We pass original IDs to reconstruct submission files later
            'id': item['id'] 
        }

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Quick test to verify alignment
    model_checkpoint = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_data_path = "data/train_data.jsonl"
    
    # Replace with your actual path
    try:
        dataset = NERDataset(train_data_path, tokenizer)
        print(f"Success! Loaded {len(dataset)} samples.")
        
        # Check one sample
        sample = dataset[0]
        print("Input IDs shape:", sample['input_ids'].shape)
        print("Labels shape:", sample['labels'].shape)
        
        # Verify the -100 masking
        print("First 10 Labels:", sample['labels'][:10])
    except FileNotFoundError:
        print("Please ensure 'train_data.jsonl' is in the directory.")