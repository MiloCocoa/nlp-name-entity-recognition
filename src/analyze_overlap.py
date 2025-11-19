import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from train_eval_bilstm_glove import BiLSTMPOSModel, BiLSTMDataset
from data_processing import NERDataset, id2label, LABEL_LIST
import spacy

# CONFIG
VAL_FILE = "data/val_split.jsonl"
DEBERTA_PATH = "models/final_model_deberta_fixed"
DISTIL_PATH = "models/final_model_distil_fixed"
BILSTM_PATH = "models/bilstm_fixed.pth"

def get_preds_transformer(model_path):
    print(f"Loading {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    dataset = NERDataset(VAL_FILE, tokenizer, max_len=128, is_test=True)
    trainer = Trainer(model=model)
    preds, _, _ = trainer.predict(dataset)
    return np.argmax(preds, axis=2), tokenizer

def get_preds_bilstm():
    print("Loading BiLSTM...")
    # Rebuild dataset to get vocab
    train_ds = BiLSTMDataset("data/train_split.jsonl", training=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPOSModel(len(train_ds.word2id), len(train_ds.pos2id), len(LABEL_LIST)).to(device)
    model.load_state_dict(torch.load(BILSTM_PATH))
    model.eval()
    
    preds = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            # Process one by one (inefficient but simple reuse of logic)
            
            # --- FIX START: Handle Empty Lines ---
            if len(row['tokens']) == 0:
                preds.append(np.array([])) # Append empty result to keep index alignment
                continue
            # --- FIX END ---

            word_ids = [train_ds.word2id.get(t, train_ds.word2id["<UNK>"]) for t in row['tokens']]
            doc = spacy.tokens.Doc(train_ds.nlp.vocab, words=row['tokens'])
            train_ds.nlp(doc)
            pos_ids = [train_ds.pos2id.get(t.pos_, train_ds.pos2id["<UNK>"]) for t in doc]
            
            w_t = torch.tensor([word_ids], dtype=torch.long).to(device)
            p_t = torch.tensor([pos_ids], dtype=torch.long).to(device)
            logits = model(w_t, p_t)
            preds.append(torch.argmax(logits, dim=2)[0].cpu().numpy())
    return preds

def align_labels(pred_ids, tokenizer, original_tokens):
    tokenized = tokenizer(original_tokens, is_split_into_words=True, truncation=True, max_length=128, padding='max_length')
    word_ids = tokenized.word_ids()
    tags = []
    prev_word = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None: continue
        if word_idx != prev_word:
            tags.append(id2label[pred_ids[idx]])
            prev_word = word_idx
    # Truncate/Pad
    if len(tags) < len(original_tokens): tags += ["O"] * (len(original_tokens) - len(tags))
    return tags[:len(original_tokens)]

def main():
    # 1. Get raw predictions
    deb_raw, deb_tok = get_preds_transformer(DEBERTA_PATH)
    dis_raw, dis_tok = get_preds_transformer(DISTIL_PATH)
    bil_raw = get_preds_bilstm()
    
    # 2. Load Truth
    ground_truth = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        ground_truth = [x['ner_tags'] for x in data]

    # 3. Analysis Loop
    agreement_count = 0
    total_tokens = 0
    oracle_correct = 0
    
    deb_correct = 0
    dis_correct = 0
    bil_correct = 0
    
    print("\nAnalyzing Overlap...")
    for i, row in enumerate(data):
        tokens = row['tokens']
        truth = row['ner_tags']
        
        # Align Transformer Predictions
        p_deb = align_labels(deb_raw[i], deb_tok, tokens)
        p_dis = align_labels(dis_raw[i], dis_tok, tokens)
        
        # BiLSTM is already word-level, just map IDs
        p_bil = [id2label[p] for p in bil_raw[i]]
        # Length safety
        p_bil = p_bil[:len(tokens)] + ["O"]*(len(tokens)-len(p_bil))
        
        # Compare
        for j, tag in enumerate(truth):
            if tag == "O": continue # Ignore O tags to focus on entities
            
            total_tokens += 1
            d_ok = (p_deb[j] == tag)
            s_ok = (p_dis[j] == tag)
            b_ok = (p_bil[j] == tag)
            
            if d_ok: deb_correct += 1
            if s_ok: dis_correct += 1
            if b_ok: bil_correct += 1
            
            # Oracle: Is at least one model right?
            if d_ok or s_ok or b_ok:
                oracle_correct += 1
                
            # Agreement: Do they all say the same thing?
            if p_deb[j] == p_dis[j] == p_bil[j]:
                agreement_count += 1

    print(f"\n--- Stats on {total_tokens} Entity Tokens (excluding 'O') ---")
    print(f"DeBERTa Accuracy:   {deb_correct/total_tokens:.2%}")
    print(f"DistilBERT Accuracy:{dis_correct/total_tokens:.2%}")
    print(f"BiLSTM Accuracy:    {bil_correct/total_tokens:.2%}")
    print(f"Full Agreement:     {agreement_count/total_tokens:.2%} (Models agree this often)")
    print(f"ORACLE Accuracy:    {oracle_correct/total_tokens:.2%} (Potential ceiling)")
    
    print("\nInterpretation:")
    gap = (oracle_correct - deb_correct) / total_tokens
    print(f"Potential Gain: +{gap:.2%} (If we pick the right model perfectly)")

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    BiLSTMDataset.nlp = nlp
    main()