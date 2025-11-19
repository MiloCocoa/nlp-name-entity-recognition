import json
import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from data_processing import NERDataset, id2label, LABEL_LIST
from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset

# CONFIG
VAL_FILE = "data/val_split.jsonl"
DEBERTA_PATH = "models/final_model_deberta_fixed"
DISTIL_PATH = "models/final_model_distil_fixed"
BILSTM_PATH = "models/bilstm_fixed.pth"

def predict_transformer(model_path, output_name):
    print(f"Predicting with {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    # Load Val set WITHOUT shuffling
    dataset = NERDataset(VAL_FILE, tokenizer, max_len=128, is_test=True) 
    
    trainer = Trainer(model=model)
    preds, _, _ = trainer.predict(dataset)
    preds = np.argmax(preds, axis=2)
    
    # Decode logic (align subwords)
    results = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    for i, row in enumerate(raw_data):
        pred_ids = preds[i]
        tokenized = tokenizer(row['tokens'], is_split_into_words=True, truncation=True, max_length=128, padding='max_length')
        word_ids = tokenized.word_ids()
        
        tags = []
        prev_word = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None: continue
            if word_idx != prev_word:
                tags.append(id2label[pred_ids[idx]])
                prev_word = word_idx
        
        # Length safety
        if len(tags) < len(row['tokens']): tags += ["O"] * (len(row['tokens']) - len(tags))
        tags = tags[:len(row['tokens'])]
        
        results.append(tags)
    
    return results

def predict_bilstm(model_path):
    print("Predicting with BiLSTM...")
    # Must rebuild dataset to recover vocab from TRAIN
    # Note: Ensure the filename matches what you have (train_bilstm_glove or train_eval_bilstm_glove)
    try:
        from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset
    except ImportError:
        # Fallback if you named it differently
        from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset

    train_ds = BiLSTMDataset("data/train_split.jsonl", training=True) # Load vocab
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPOSModel(len(train_ds.word2id), len(train_ds.pos2id), len(LABEL_LIST)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    results = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            tokens = row['tokens']
            
            # --- FIX: Handle empty lines ---
            if len(tokens) == 0:
                results.append([])
                continue
            # -------------------------------
            
            # Preprocess
            word_ids = [train_ds.word2id.get(t, train_ds.word2id["<UNK>"]) for t in tokens]
            doc = spacy.tokens.Doc(train_ds.nlp.vocab, words=tokens)
            train_ds.nlp(doc)
            pos_ids = [train_ds.pos2id.get(t.pos_, train_ds.pos2id["<UNK>"]) for t in doc]
            
            w_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
            p_tensor = torch.tensor([pos_ids], dtype=torch.long).to(device)
            
            logits = model(w_tensor, p_tensor)
            preds = torch.argmax(logits, dim=2)[0]
            tags = [id2label[p.item()] for p in preds]
            results.append(tags)
    return results

def main():
    # 1. Generate Predictions
    preds_deberta = predict_transformer(DEBERTA_PATH, "temp")
    preds_distil = predict_transformer(DISTIL_PATH, "temp")
    preds_bilstm = predict_bilstm(BILSTM_PATH)
    
    # 2. Ensemble Logic (Weighted Voting)
    print("Calculating Ensemble Metrics...")
    
    final_preds = []
    # Weights: DeBERTa=3, Distil=1, BiLSTM=1
    weights = [3, 1, 1] 
    
    from collections import Counter
    for i in range(len(preds_deberta)):
        sentence_preds = []
        length = len(preds_deberta[i])
        
        for j in range(length):
            c = Counter()
            # Vote
            if j < len(preds_deberta[i]): c[preds_deberta[i][j]] += weights[0]
            if j < len(preds_distil[i]): c[preds_distil[i][j]] += weights[1]
            if j < len(preds_bilstm[i]): c[preds_bilstm[i][j]] += weights[2]
            
            sentence_preds.append(c.most_common(1)[0][0])
        final_preds.append(sentence_preds)

    # 3. Evaluate against Ground Truth
    from seqeval.metrics import classification_report
    
    true_labels = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            true_labels.append(json.loads(line)['ner_tags'])
            
    print("\n" + "="*30)
    print("ENSEMBLE MODEL RESULTS (VALIDATION)")
    print("="*30)
    print(classification_report(true_labels, final_preds, mode='strict'))
    print("="*30)

if __name__ == "__main__":
    # Hack for spacy in BiLSTM class
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    BiLSTMDataset.nlp = nlp
    main()