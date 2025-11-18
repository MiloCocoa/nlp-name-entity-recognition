import numpy as np
import evaluate
from seqeval.metrics import classification_report
from data_processing import id2label # Import from your local file

# Load seqeval (Standard for BIO tagging)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """
    Transforms logits -> labels, removes -100 padding, and calculates strict F1.
    """
    predictions, labels = p
    
    # Convert logits to class IDs
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute strict metrics
    results = seqeval.compute(predictions=true_predictions, references=true_labels, mode='strict')

    # PRINT THE REPORT for your PDF submission
    print("\n" + "="*30)
    print("ENTITY SPAN CLASSIFICATION REPORT")
    print("="*30)
    # This output matches the PDF's "Correct Evaluation" table requirement [cite: 56]
    print(classification_report(true_labels, true_predictions, mode='strict'))
    print("="*30 + "\n")

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }