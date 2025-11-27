"""
Evaluation Script for Fine-Tuned CodeBERT Plagiarism Detector
Evaluates the model on the held-out test dataset
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import Dataset
import evaluate
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


MAX_LENGTH = 255


def tokenization(row, tokenizer):
    """Tokenize code pairs"""
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=MAX_LENGTH,
    )
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    return tokenized_inputs


def load_test_dataset(data_path: Path, tokenizer):
    """Load and tokenize test dataset"""
    print(f"Loading test dataset from {data_path}")
    df= pd.read_csv(data_path)
    
    dataset = Dataset.from_pandas(df)
    
    # Rename label column if needed
    if "semantic_clone" in dataset.column_names:
        dataset = dataset.rename_column("semantic_clone", "label")
    
    # Tokenize
    dataset = dataset.map(
        lambda row: tokenization(row, tokenizer),
        batched=False
    )
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    print(f"Test dataset size: {len(dataset)}")
    return dataset


def evaluate_model(model_path: Path, test_dataset, device):
    """
    Evaluate the model on test dataset
    """
    print(f"Loading model from {model_path}")
    model = RobertaForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    # Prepare for evaluation
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            label = item['label'].item()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            
            all_predictions.append(prediction)
            all_labels.append(label)
            all_probs.append(probs.cpu().numpy()[0])
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def compute_detailed_metrics(predictions, labels):
    """Compute detailed evaluation metrics"""
    
    # Load evaluation metrics
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    
    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }
    
    # Classification report
    report = classification_report(labels, predictions, target_names=['Not Plagiarism', 'Plagiarism'])
    
    return metrics, report


def plot_confusion_matrix(labels, predictions, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Plagiarism', 'Plagiarism'],
                yticklabels=['Not Plagiarism', 'Plagiarism'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeBERT plagiarism detector")
    parser.add_argument("--model_path", type=str, 
                       default="models/plagiarism_detector/checkpoint-1221",
                       help="Path to trained model")
    parser.add_argument("--test_data", type=str, default="data/test_dataset.csv",
                       help="Path to test dataset CSV")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--use_cpu", action="store_true",
                       help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Setup
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cpu" if args.use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    
    # Load test dataset
    test_dataset = load_test_dataset(test_data_path, tokenizer)
    
    # Evaluate
    predictions, labels, probs = evaluate_model(model_path, test_dataset, device)
    
    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    
    metrics, report = compute_detailed_metrics(predictions, labels)
    
    print("Overall Metrics:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(labels, predictions, cm_path)
    
    # Compare with benchmarks
    print("\n" + "="*80)
    print("COMPARISON WITH BENCHMARKS")
    print("="*80)

    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Success check
    if metrics['f1'] >= 0.88:
        print("\n✓ Model meets minimum performance threshold (F1 >= 0.88)")
    else:
        print("\n✗ Model below threshold. Consider:")
        print("  - Training for more epochs")
        print("  - Using a larger dataset")
        print("  - Adjusting hyperparameters")


if __name__ == "__main__":
    main()
