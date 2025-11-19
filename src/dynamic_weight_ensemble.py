import json
import torch
import numpy as np
import spacy
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from data_processing import NERDataset, id2label, LABEL_LIST
from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset
from seqeval.metrics import classification_report

# CONFIG
VAL_FILE = "data/val_split.jsonl"
DEBERTA_PATH = "models/final_model_deberta_fixed"
DISTIL_PATH = "models/final_model_distil_fixed"
BILSTM_PATH = "models/bilstm_fixed.pth"

# --- DYNAMIC WEIGHTS TABLE (Derived from your Validation Reports) ---
# Format: "TAG": [DeBERTa_Precision, Distil_Precision, BiLSTM_Precision]
# We use Precision because we want to know: "If model says X, how likely is it really X?"
PRECISION_SCORES = {
    "B-Artist":          [0.84, 0.80, 0.77],
    "I-Artist":          [0.84, 0.80, 0.77], # Assuming I-tags track B-tags
    
    "B-Facility":        [0.81, 0.80, 0.72],
    "I-Facility":        [0.81, 0.80, 0.72],
    
    "B-HumanSettlement": [0.93, 0.93, 0.88],
    "I-HumanSettlement": [0.93, 0.93, 0.88],
    
    "B-ORG":             [0.83, 0.79, 0.72],
    "I-ORG":             [0.83, 0.79, 0.72],
    
    "B-OtherPER":        [0.61, 0.66, 0.57], # DistilBERT is the expert here!
    "I-OtherPER":        [0.61, 0.66, 0.57],
    
    "B-Politician":      [0.77, 0.72, 0.68],
    "I-Politician":      [0.77, 0.72, 0.68],
    
    "B-PublicCorp":      [0.72, 0.73, 0.69], # DistilBERT wins here too
    "I-PublicCorp":      [0.72, 0.73, 0.69],
    
    "O":                 [0.99, 0.99, 0.99]  # Everyone is good at O
}

def get_preds_transformer(model_path):
    print(f"Predicting {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    dataset = NERDataset(VAL_FILE, tokenizer, max_len=128, is_test=True)
    trainer = Trainer(model=model)
    preds, _, _ = trainer.predict(dataset)
    preds = np.argmax(preds, axis=2)
    
    results = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            row = json.loads(line)
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
            if len(tags) < len(row['tokens']): tags += ["O"] * (len(row['tokens']) - len(tags))
            results.append(tags[:len(row['tokens'])])
    return results

def get_preds_bilstm():
    print("Predicting BiLSTM...")
    try:
        from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset
    except ImportError:
        from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset

    train_ds = BiLSTMDataset("data/train_split.jsonl", training=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPOSModel(len(train_ds.word2id), len(train_ds.pos2id), len(LABEL_LIST)).to(device)
    model.load_state_dict(torch.load(BILSTM_PATH))
    model.eval()
    
    results = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if len(row['tokens']) == 0:
                results.append([])
                continue
                
            word_ids = [train_ds.word2id.get(t, train_ds.word2id["<UNK>"]) for t in row['tokens']]
            doc = spacy.tokens.Doc(train_ds.nlp.vocab, words=row['tokens'])
            train_ds.nlp(doc)
            pos_ids = [train_ds.pos2id.get(t.pos_, train_ds.pos2id["<UNK>"]) for t in doc]
            
            w_t = torch.tensor([word_ids], dtype=torch.long).to(device)
            p_t = torch.tensor([pos_ids], dtype=torch.long).to(device)
            
            logits = model(w_t, p_t)
            preds = torch.argmax(logits, dim=2)[0]
            results.append([id2label[p.item()] for p in preds])
    return results

def main():
    # 1. Get Predictions
    p_deb = get_preds_transformer(DEBERTA_PATH)
    p_dis = get_preds_transformer(DISTIL_PATH)
    p_bil = get_preds_bilstm()
    
    # 2. Dynamic Voting Logic
    print("Running Dynamic Ensemble...")
    final_preds = []
    
    for i in range(len(p_deb)):
        sent_preds = []
        # Length safety check
        length = min(len(p_deb[i]), len(p_dis[i]), len(p_bil[i]))
        
        for j in range(length):
            # Candidates from each model
            tag_d = p_deb[i][j]
            tag_s = p_dis[i][j]
            tag_b = p_bil[i][j]
            
            # Initialize voting bucket
            votes = Counter()
            
            # --- DYNAMIC WEIGHTING ---
            # Weight = Precision^2 (Squaring emphasizes high confidence)
            
            # DeBERTa Vote
            w_d = PRECISION_SCORES.get(tag_d, [0.5, 0, 0])[0] ** 2
            votes[tag_d] += w_d
            
            # DistilBERT Vote
            w_s = PRECISION_SCORES.get(tag_s, [0, 0.5, 0])[1] ** 2
            votes[tag_s] += w_s
            
            # BiLSTM Vote
            w_b = PRECISION_SCORES.get(tag_b, [0, 0, 0.5])[2] ** 2
            votes[tag_b] += w_b
            
            # Winner takes all
            best_tag = votes.most_common(1)[0][0]
            sent_preds.append(best_tag)
        final_preds.append(sent_preds)

    # 3. Evaluate
    true_labels = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            true_labels.append(json.loads(line)['ner_tags'])
            
    print("\n" + "="*30)
    print("DYNAMIC ENSEMBLE RESULTS")
    print("="*30)
    print(classification_report(true_labels, final_preds, mode='strict'))
    print("="*30)

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    try:
        from train_eval_bilstm_glove import BiLSTMDataset
        BiLSTMDataset.nlp = nlp
    except: pass
    main()