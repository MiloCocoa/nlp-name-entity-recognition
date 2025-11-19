import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import spacy
import os
from seqeval.metrics import classification_report
from data_processing import id2label, LABEL_LIST, label2id

# --- CONFIGURATION ---
BATCH_SIZE = 32
EMBED_DIM = 100     # Must match GloVe dimension (glove.6B.100d.txt)
HIDDEN_DIM = 256
POS_EMBED_DIM = 25
EPOCHS = 10         # Increased epochs since pre-trained needs less "learning" but more "tuning"
LEARNING_RATE = 0.001
SEED = 42
GLOVE_PATH = "data/glove.6B.100d.txt" # Ensure this file exists in your path!

# Load Spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def load_glove_embeddings(path, word2id, embed_dim):
    """
    Parses GloVe text file and creates an embedding matrix matching word2id.
    """
    print(f"Loading GloVe vectors from {path}...")
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Initialize embedding matrix with random values (for OOV words)
    # We use a normal distribution to match GloVe's scale roughly
    vocab_size = len(word2id)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))
    
    hits = 0
    misses = 0
    
    for word, i in word2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # Try lower case
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                
    print(f"GloVe Loaded: {hits} hits, {misses} misses. (Coverage: {hits/vocab_size:.1%})")
    return torch.tensor(embedding_matrix, dtype=torch.float32)

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
    def __init__(self, vocab_size, pos_vocab_size, num_labels, pretrained_embeddings=None):
        super().__init__()
        
        # 1. Word Embeddings
        if pretrained_embeddings is not None:
            # Initialize with GloVe
            self.word_embed = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
            print("Model initialized with Pretrained Embeddings.")
        else:
            # Fallback to random
            self.word_embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        # 2. POS Embeddings
        self.pos_embed = nn.Embedding(pos_vocab_size, POS_EMBED_DIM, padding_idx=0)
        
        # 3. LSTM & Classifier
        self.lstm = nn.LSTM(EMBED_DIM + POS_EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5) # Added dropout for regularization
        self.classifier = nn.Linear(HIDDEN_DIM * 2, num_labels)
        
    def forward(self, word_ids, pos_ids):
        x_w = self.word_embed(word_ids)
        x_p = self.pos_embed(pos_ids)
        x = torch.cat([x_w, x_p], dim=2)
        
        x = self.dropout(x) # Apply dropout before LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out) # And after
        
        return self.classifier(lstm_out)

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("Loading data...")
    full_ds = BiLSTMDataset("data/train_data.jsonl", training=True)
    
    # Deterministic Split
    generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # --- LOAD GLOVE ---
    if os.path.exists(GLOVE_PATH):
        embedding_matrix = load_glove_embeddings(GLOVE_PATH, full_ds.word2id, EMBED_DIM)
    else:
        print(f"WARNING: '{GLOVE_PATH}' not found. Using random embeddings.")
        embedding_matrix = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    model = BiLSTMPOSModel(
        vocab_size=len(full_ds.word2id),
        pos_vocab_size=len(full_ds.pos2id),
        num_labels=len(LABEL_LIST),
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
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

    # --- EVALUATION ---
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
    print("BiLSTM + POS + GloVe RESULTS")
    print("="*30)
    print(classification_report(true_labels, pred_labels, mode='strict'))
    print("="*30)
    
    torch.save(model.state_dict(), "bilstm_glove_model.pth")

if __name__ == "__main__":
    main()