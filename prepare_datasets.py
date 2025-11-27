"""
Dataset Preparation Script for CodeBERT Fine-Tuning
Creates consolidated CSV files from GPTCloneBench and Semantic Benchmark datasets
"""

import os
import pandas as pd
from pathlib import Path
import re
from typing import List, Tuple

def process_semantic_benchmark(base_path: Path) -> pd.DataFrame:
    """
    Process Semantic Benchmark dataset to create clone pairs
    
    Args:
        base_path: Path to Semantic Benchmark folder
        
    Returns:
        DataFrame with columns: clone1, clone2, semantic_clone
    """
    print("Processing Semantic Benchmark dataset...")
    
    # Path to Python standalone clones
    python_path = base_path / "Python"
    
    if not python_path.exists():
        raise FileNotFoundError(f"Python folder not found at {python_path}")
    
    # Find all Python files
    python_files = list(python_path.glob("**/*.py"))
    print(f"Found {len(python_files)} Python files")
    
    # Pattern to extract clone ID from filename
    pattern = re.compile(r"Clone(\d+)\.py")
    method_name_pattern = re.compile(r"def\s+(.+?)\(")
    
    clones = []
    for file in python_files:
        match = pattern.search(str(file))
        if match:
            clone_id = match.group(1)
            
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split into two clone pairs (separated by triple newlines)
            parts = content.split("\n\n\n")
            if len(parts) >= 2:
                clone1, clone2 = parts[0], parts[1]
                
                # Remove method names for anonymization (replace with generic name)
                clone1 = method_name_pattern.sub("def method_name(", clone1.strip(), count=1)
                clone2 = method_name_pattern.sub("def method_name(", clone2.strip(), count=1)
                
                clones.append({
                    'clone1': clone1,
                    'clone2': clone2,
                    'semantic_clone': 1  # These are true clones
                })
    
    df = pd.DataFrame(clones)
    print(f"Extracted {len(df)} clone pairs")
    
    # Create dissimilar pairs by mixing first half with second half
    if len(df) > 100:
        split_point = len(df) // 2
        first_half = df.iloc[:split_point]
        second_half = df.iloc[split_point:]
        
        dissimilar_pairs = []
        min_len = min(len(first_half), len(second_half))
        
        for i in range(min_len):
            dissimilar_pairs.append({
                'clone1': first_half.iloc[i]['clone1'],
                'clone2': second_half.iloc[i]['clone1'],
                'semantic_clone': 0
            })
        
        dissimilar_df = pd.DataFrame(dissimilar_pairs)
        print(f"Created {len(dissimilar_df)} dissimilar pairs")
        
        # Merge
        merged_df = pd.concat([df, dissimilar_df], ignore_index=True)
    else:
        merged_df = df
    
    print(f"Total Semantic Benchmark samples: {len(merged_df)}")
    return merged_df


def process_gptclonebench(base_path: Path, languages: List[str] = ['py']) -> pd.DataFrame:
    """
    Process GPTCloneBench dataset to create clone pairs
    
    Args:
        base_path: Path to GPTCloneBench folder
        languages: List of languages to process ('py', 'java', 'c', 'cs')
        
    Returns:
        DataFrame with columns: clone1, clone2, semantic_clone
    """
    print("Processing GPTCloneBench dataset...")
    
    standalone_path = base_path / "standalone"
    
    if not standalone_path.exists():
        raise FileNotFoundError(f"Standalone folder not found at {standalone_path}")
    
    positive_pairs = []
    all_snippets = []
    
    # Helper function to process files
    def process_files_in_path(path_to_scan):
        count = 0
        if not path_to_scan.exists():
            return 0
            
        # Find all code files recursively
        for lang in languages:
            code_files = list(path_to_scan.glob(f"**/*.{lang}"))
            
            for file in code_files:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Split into clone pairs
                    parts = content.split("\n\n\n")
                    if len(parts) >= 2:
                        c1 = parts[0].strip()
                        c2 = parts[1].strip()
                        
                        if c1 and c2:
                            positive_pairs.append({
                                'clone1': c1,
                                'clone2': c2,
                                'semantic_clone': 1  # ALL clones are plagiarism
                            })
                            all_snippets.append(c1)
                            all_snippets.append(c2)
                            count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        return count

    # 1. Process true semantic clones (Type 3/4)
    print("Processing True Semantic Clones (Type 3/4)...")
    true_clones_path = standalone_path / "true_semantic_clones"
    n_true = process_files_in_path(true_clones_path)
    print(f"  Found {n_true} pairs")

    # 2. Process false semantic clones (Type 1/2) -> NOW LABELED AS 1 (PLAGIARISM)
    print("Processing 'False' Semantic Clones (Type 1/2 - Syntactic)...")
    false_clones_path = standalone_path / "false_semantic_clones"
    n_false = process_files_in_path(false_clones_path)
    print(f"  Found {n_false} pairs")
    
    print(f"Total Positive Pairs: {len(positive_pairs)}")
    
    # 3. Generate Negative Pairs (Randomly pair different snippets)
    print("Generating Negative Pairs (Non-Plagiarism)...")
    import random
    random.seed(42)
    
    negative_pairs = []
    num_negatives = len(positive_pairs)  # Aim for 50/50 balance
    
    # Deduplicate snippets to avoid exact same code being paired
    unique_snippets = list(set(all_snippets))
    print(f"  Unique code snippets available: {len(unique_snippets)}")
    
    if len(unique_snippets) < 2:
        print("Warning: Not enough snippets to generate negatives")
        return pd.DataFrame(positive_pairs)
        
    attempts = 0
    while len(negative_pairs) < num_negatives and attempts < num_negatives * 5:
        attempts += 1
        s1 = random.choice(unique_snippets)
        s2 = random.choice(unique_snippets)
        
        # Simple heuristic: if lengths differ significantly or strings are diff, assume diff
        # Ideally we check if they are NOT in positive_pairs, but that's O(N^2)
        # For now, just ensure they aren't identical string
        if s1 != s2:
            negative_pairs.append({
                'clone1': s1,
                'clone2': s2,
                'semantic_clone': 0
            })
            
    print(f"  Generated {len(negative_pairs)} negative pairs")
    
    # Combine
    all_data = positive_pairs + negative_pairs
    df = pd.DataFrame(all_data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total GPTCloneBench samples: {len(df)}")
    print(f"  - Plagiarism (1): {(df['semantic_clone'] == 1).sum()}")
    print(f"  - Non-Plagiarism (0): {(df['semantic_clone'] == 0).sum()}")
    
    return df


def create_datasets(output_dir: Path):
    """
    Main function to create all datasets
    """
    base_dir = Path(__file__).parent / "0_dataset_creations"
    
    # Process Semantic Benchmark
    semantic_path = base_dir / "Semantic Benchmark"
    if semantic_path.exists():
        semantic_df = process_semantic_benchmark(semantic_path)
        output_file = output_dir / "semantic_benchmark_dataset.csv"
        semantic_df.to_csv(output_file, index=False)
        print(f"Saved Semantic Benchmark dataset to {output_file}")
    else:
        print("Semantic Benchmark folder not found, skipping")
        semantic_df = None
    
    # Process GPTCloneBench
    gpt_path = base_dir / "GPTCloneBench"
    if gpt_path.exists():
        gpt_df = process_gptclonebench(gpt_path, languages=['py'])
        output_file = output_dir / "gptclonebench_dataset.csv"
        gpt_df.to_csv(output_file, index=False)
        print(f"Saved GPTCloneBench dataset to {output_file}")
    else:
        print("GPTCloneBench folder not found, skipping")
        gpt_df = None
    
    # Create combined dataset
    if semantic_df is not None and gpt_df is not None:
        combined_df = pd.concat([semantic_df, gpt_df], ignore_index=True)
        output_file = output_dir / "combined_dataset.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved combined dataset to {output_file}")
        print(f"Total samples: {len(combined_df)}")
        print(f"  - True clones: {(combined_df['semantic_clone'] == 1).sum()}")
        print(f"  - False clones: {(combined_df['semantic_clone'] == 0).sum()}")


if __name__ == "__main__":
    # Create data directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    create_datasets(output_dir)
    print("\nDataset preparation complete!")
