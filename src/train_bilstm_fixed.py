import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset, collate_fn, load_glove_embeddings
from data_processing import LABEL_LIST
import os

# CONFIG
BATCH_SIZE = 32
EMBED_DIM = 100
POS_EMBED_DIM = 25
HIDDEN_DIM = 256
EPOCHS = 10
LR = 0.001
GLOVE_PATH = "data/glove.6B.100d.txt"
OUTPUT_FILE = "bilstm_fixed.pth"

def main():
    print("Loading Fixed Splits...")
    # Important: Build vocab from TRAIN only to avoid leakage, 
    # but handle UNKs in Val gracefully (the dataset class handles this)
    train_ds = BiLSTMDataset("data/train_split.jsonl", training=True)
    val_ds = BiLSTMDataset("data/val_split.jsonl", word2id=train_ds.word2id, pos2id=train_ds.pos2id, training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Load GloVe
    if os.path.exists(GLOVE_PATH):
        embedding_matrix = load_glove_embeddings(GLOVE_PATH, train_ds.word2id, EMBED_DIM)
    else:
        embedding_matrix = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPOSModel(len(train_ds.word2id), len(train_ds.pos2id), len(LABEL_LIST), embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
        
    torch.save(model.state_dict(), OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()