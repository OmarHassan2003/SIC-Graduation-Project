"""
CodeBERT Fine-Tuning Script for Code Plagiarism Detection
Based on CodeCloneBERT implementation
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
import argparse
import json
from datetime import datetime


# Constants
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 255


def tokenization(row, tokenizer):
    """
    Tokenize code pairs following the repository's approach
    Uses text_pair format to concatenate code snippets
    """
    # Ensure text inputs are strings
    code1 = str(row["clone1"]) if pd.notna(row["clone1"]) else ""
    code2 = str(row["clone2"]) if pd.notna(row["clone2"]) else ""
    
    tokenized_inputs = tokenizer(
        code1,
        code2,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=MAX_LENGTH,
    )
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    return tokenized_inputs


def load_and_prepare_dataset(data_path: Path, tokenizer):
    """
    Load dataset from CSV and prepare for training
    
    Returns train, validation, and test datasets
    """
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Remove rows with NaN values in important columns
    initial_size = len(df)
    df = df.dropna(subset=['clone1', 'clone2', 'semantic_clone'])

    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['semantic_clone'].value_counts()}")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # Rename label column if needed
    if "semantic_clone" in dataset.column_names:
        dataset = dataset.rename_column("semantic_clone", "label")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    dataset = dataset.map(
        lambda row: tokenization(row, tokenizer),
        batched=False
    )
    
    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Create splits
    print("Creating train/val/test splits...")
    dataset = dataset.shuffle(seed=42)
    
    # 70% train, 15% validation, 15% test
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    train_dataset = train_test_split['train']
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Save test dataset for later evaluation
    test_dataset.to_csv("data/test_dataset.csv")
    print("Saved test dataset to data/test_dataset.csv")
    
    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred, accuracy, precision, recall, f1):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"]
    }


def train_model(
    train_dataset,
    val_dataset,
    output_dir: Path,
    batch_size: int = 16,
    num_epochs: int = 30,
    learning_rate: float = 2e-5,
    use_cpu: bool = False
):
    """
    Train the CodeBERT model for plagiarism detection
    """
    
    # Setup device
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    ).to(device)
    
    # Load metrics
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        adam_epsilon=1e-8,
        num_train_epochs=num_epochs,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir=str(output_dir),
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=3,
        use_cpu=use_cpu,
        report_to="none",  # Disable wandb
        fp16=not use_cpu and torch.cuda.is_available(),  # Use mixed precision on GPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, accuracy, precision, recall, f1
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80 + "\n")
    
    # Save model
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    print(f"Model saved to {final_model_dir}")
    
    # Save training metrics
    metrics = train_result.metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_file}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train CodeBERT for plagiarism detection")
    parser.add_argument("--data_path", type=str, default="data/combined_dataset.csv",
                       help="Path to dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="models/plagiarism_detector",
                       help="Output directory for model and logs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--use_cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Load and prepare datasets
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(
        data_path, tokenizer
    )
    
    # Train model
    trainer = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_cpu=args.use_cpu
    )
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set:")
    print("="*80)
    val_metrics = trainer.evaluate()
    print(f"Validation Metrics:")
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save validation metrics
    val_metrics_file = output_dir / "validation_metrics.json"
    with open(val_metrics_file, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    print(f"\nAll outputs saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Run evaluate_model.py to evaluate on the test set")
    print("  2. Run streamlit_app.py to test the model interactively")


if __name__ == "__main__":
    main()
