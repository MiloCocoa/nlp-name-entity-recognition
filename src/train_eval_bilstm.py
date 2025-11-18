import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import spacy
from seqeval.metrics import classification_report # Use seqeval for report
from data_processing import id2label, LABEL_LIST, label2id # Reuse existing config

# --- CONFIGURATION ---
BATCH_SIZE = 32
EMBED_DIM = 100
HIDDEN_DIM = 256
POS_EMBED_DIM = 25
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 42 # Fix the seed for reproducibility

# Load Spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class BiLSTMDataset(Dataset):
    def __init__(self, data_path, word2id=None, pos2id=None, training=True):
        self.data = []
        self.word2id = word2id if word2id else {"<PAD>": 0, "<UNK>": 1}
        self.pos2id = pos2id if pos2id else {"<PAD>": 0, "<UNK>": 1}
        
        if training:
            print("Building vocabulary...")
            self._build_vocab(data_path)
            
        self._load_data(data_path)

    def _build_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                for token in row['tokens']:
                    if token not in self.word2id:
                        self.word2id[token] = len(self.word2id)
                # Simple POS tagging
                doc = spacy.tokens.Doc(nlp.vocab, words=row['tokens'])
                nlp(doc)
                for token in doc:
                    if token.pos_ not in self.pos2id:
                        self.pos2id[token.pos_] = len(self.pos2id)

    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                tokens = row['tokens']
                labels = row.get('ner_tags', ["O"] * len(tokens))
                
                word_ids = [self.word2id.get(t, self.word2id["<UNK>"]) for t in tokens]
                
                doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
                nlp(doc)
                pos_ids = [self.pos2id.get(t.pos_, self.pos2id["<UNK>"]) for t in doc]
                
                label_ids = [label2id.get(l, 0) for l in labels]
                
                self.data.append({
                    "word_ids": torch.tensor(word_ids, dtype=torch.long),
                    "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    word_ids = pad_sequence([item['word_ids'] for item in batch], batch_first=True, padding_value=0)
    pos_ids = pad_sequence([item['pos_ids'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    return word_ids, pos_ids, labels

class BiLSTMPOSModel(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, num_labels):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos_embed = nn.Embedding(pos_vocab_size, POS_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM + POS_EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, num_labels)
        
    def forward(self, word_ids, pos_ids):
        x = torch.cat([self.word_embed(word_ids), self.pos_embed(pos_ids)], dim=2)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out)

def main():
    # Set Seed for Reproducibility
    torch.manual_seed(SEED)
    
    print("Loading data...")
    full_ds = BiLSTMDataset("data/train_data.jsonl", training=True)
    
    # Deterministic Split using Generator
    generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPOSModel(len(full_ds.word2id), len(full_ds.pos2id), len(LABEL_LIST)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for word_ids, pos_ids, labels in train_loader:
            word_ids, pos_ids, labels = word_ids.to(device), pos_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(word_ids, pos_ids)
            loss = criterion(logits.view(-1, len(LABEL_LIST)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    # --- FINAL EVALUATION (No Leakage) ---
    print("\nRunning Final Evaluation on Validation Set...")
    model.eval()
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for word_ids, pos_ids, labels in val_loader:
            word_ids, pos_ids, labels = word_ids.to(device), pos_ids.to(device), labels.to(device)
            logits = model(word_ids, pos_ids)
            preds = torch.argmax(logits, dim=2)
            
            for i in range(len(labels)):
                p_seq, l_seq = [], []
                for j in range(len(labels[i])):
                    if labels[i][j].item() != -100:
                        p_seq.append(id2label[preds[i][j].item()])
                        l_seq.append(id2label[labels[i][j].item()])
                pred_labels.append(p_seq)
                true_labels.append(l_seq)

    print("\n" + "="*30)
    print("BiLSTM + POS RESULTS (VALID)")
    print("="*30)
    print(classification_report(true_labels, pred_labels, mode='strict'))
    print("="*30)
    
    torch.save(model.state_dict(), "bilstm_pos_model.pth")

if __name__ == "__main__":
    main()