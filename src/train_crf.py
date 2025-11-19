import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torchcrf import CRF
from data_processing import NERDataset, LABEL_LIST, id2label, label2id
from seqeval.metrics import classification_report
import numpy as np
import os

# --- CONFIGURATION ---
# Start with DistilBERT for speed, switch to "microsoft/deberta-v3-large" for high score
MODEL_CHECKPOINT = "distilbert-base-uncased" 
BATCH_SIZE = 16         # Lower to 4 if using DeBERTa-Large
EPOCHS = 4
LR = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Project BERT hidden size (768 for base, 1024 for large) to num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # The CRF Layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 2. Project to Tag Space (Emissions)
        emissions = self.classifier(sequence_output) # [Batch, SeqLen, NumLabels]
        
        # 3. Compute Loss or Decode
        # FIX: Use attention_mask for CRF, ensuring the first token [CLS] is always masked=1
        crf_mask = attention_mask.byte()
        
        if labels is not None:
            # FIX: Replace -100 (ignore index) with 0 ("O")
            # The CRF needs a valid path for every token in the mask. 
            # We teach it that [CLS], [SEP], and sub-words are "O".
            safe_labels = labels.clone()
            safe_labels[labels == -100] = 0 
            
            # Calculate negative log likelihood
            log_likelihood = self.crf(emissions, safe_labels, mask=crf_mask, reduction='mean')
            return -log_likelihood
        else:
            # Prediction (Decode)
            return self.crf.decode(emissions, mask=crf_mask)

def main():
    print(f"Initializing Transformer-CRF with {MODEL_CHECKPOINT}...")
    
    # 1. Load Data
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
        
    full_ds = NERDataset("data/train_data.jsonl", tokenizer, max_len=MAX_LEN)
    
    # 90/10 Split
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)
    
    # 2. Model Setup
    model = TransformerCRF(MODEL_CHECKPOINT, len(LABEL_LIST)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Scheduler
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            loss = model(input_ids, mask, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
        
        # 4. Validation (Strict Span)
        model.eval()
        true_labels = []
        pred_labels = []
        
        print("Validating...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                # Get CRF predictions (List of Lists of Ints)
                preds = model(input_ids, mask) 
                
                # Align and Decode
                for i in range(len(labels)):
                    pred_seq = preds[i] # Matches attention_mask length
                    
                    p_list = []
                    l_list = []
                    
                    pred_idx = 0
                    for j in range(len(labels[i])):
                        label_id = labels[i][j].item()
                        
                        # Check bounds
                        if pred_idx >= len(pred_seq): break
                        
                        # If original label was NOT padding/special (-100), it's a real word
                        # We grab the prediction corresponding to this position
                        if label_id != -100:
                            p_tag = id2label[pred_seq[pred_idx]]
                            l_tag = id2label[label_id]
                            p_list.append(p_tag)
                            l_list.append(l_tag)
                        
                        # Advance prediction index if this token was in the CRF mask
                        if mask[i][j] == 1:
                            pred_idx += 1
                            
                    pred_labels.append(p_list)
                    true_labels.append(l_list)

        # Report
        print("\n" + "="*30)
        print(f"Epoch {epoch+1} Results ({MODEL_CHECKPOINT} + CRF)")
        print("="*30)
        print(classification_report(true_labels, pred_labels, mode='strict'))
        print("="*30)
        
    # Save
    torch.save(model.state_dict(), "transformer_crf_model.pth")
    print("Model Saved!")

if __name__ == "__main__":
    main()