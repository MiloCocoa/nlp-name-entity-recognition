import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from data_processing import LABEL_LIST

# --- CONFIGURATION ---
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
NUM_SAMPLES = 5  # Small sample for report feasibility
SEED = 42

def format_prompt(tokens, examples=[]):
    """
    Constructs a few-shot prompt for the LLM.
    """
    sentence = " ".join(tokens)
    
    # Few-shot examples (Hardcoded from train set for consistency)
    example_text = ""
    if examples:
        for ex in examples:
            ex_tokens = " ".join(ex['tokens'])
            ex_tags = " ".join(ex['ner_tags'])
            example_text += f"Sentence: {ex_tokens}\nTags: {ex_tags}\n\n"

    # The Instruction
    prompt = f"""<|user|>
You are an expert Named Entity Recognition system. 
Task: Assign BIO tags to the words in the sentence.
Labels allowed: {", ".join(LABEL_LIST)}

Format: Output ONLY the list of tags separated by spaces. The number of tags must match the number of words.

{example_text}Sentence: {sentence}
Tags:<|end|>
<|assistant|>"""
    return prompt

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Load Data
    with open("data/train_data.jsonl", 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    
    # 90/10 Split again to get the SAME validation set
    random.seed(SEED)
    random.shuffle(lines)
    split_idx = int(0.9 * len(lines))
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]
    
    # Pick 3 fixed examples for Few-Shot
    few_shot_examples = train_data[:3]
    
    # Pick subset of validation to test
    test_subset = val_data[:NUM_SAMPLES]
    
    print(f"Running Few-Shot Inference on {NUM_SAMPLES} samples...")
    
    correct_spans = 0
    total_spans = 0
    
    for i, row in enumerate(test_subset):
        prompt = format_prompt(row['tokens'], few_shot_examples)
        
        output = pipe(
            prompt, 
            max_new_tokens=512, 
            return_full_text=False,
            do_sample=False, # Deterministic
            use_cache=False
        )
        
        generated_text = output[0]['generated_text'].strip()
        pred_tags = generated_text.split()
        
        # Basic alignment (truncate or pad)
        if len(pred_tags) > len(row['tokens']):
            pred_tags = pred_tags[:len(row['tokens'])]
        elif len(pred_tags) < len(row['tokens']):
            pred_tags += ["O"] * (len(row['tokens']) - len(pred_tags))
            
        # Quick & Dirty Metric for the console (Full metric in report)
        # Just checking if we got any entities right
        print(f"\nSample {i+1}:")
        print(f"True: {row['ner_tags']}")
        print(f"Pred: {pred_tags}")
        
    print("\nDone! Review the console output to write your qualitative analysis.")

if __name__ == "__main__":
    main()