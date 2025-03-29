import pandas as pd
import os
import argparse
import numpy as np

def load_annotations(annotation_dir):
    """
    Load annotations from CSV files in the specified directory.
    
    Args:
        annotation_dir (str): Directory path containing annotations in CSV format
        
    Returns:
        dict: Dictionary mapping file IDs to their sentence boundaries
    """
    annotations = {}
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(annotation_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_id = csv_file.replace('.csv', '')
        file_path = os.path.join(annotation_dir, csv_file)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract sentence boundaries
        sentence_boundaries = []
        for _, row in df.iterrows():
            if row['sentence_type'] == 'Sentence':
                start = row['sentence_start']
                end = row['sentence_end']
                sentence_boundaries.append((start, end))
        
        annotations[file_id] = sentence_boundaries
    
    return annotations

def calculate_metrics(gold_boundaries, pred_boundaries):
    """
    Calculate precision, recall, and F1 score for sentence boundaries.
    
    Args:
        gold_boundaries (list): List of (start, end) tuples for gold annotations
        pred_boundaries (list): List of (start, end) tuples for predictions
        
    Returns:
        tuple: (precision, recall, f1_score)
    """
    # Convert boundaries to sets for easier comparison
    gold_set = set(gold_boundaries)
    pred_set = set(pred_boundaries)
    
    # Calculate true positives (boundaries that exist in both sets)
    true_positives = len(gold_set.intersection(pred_set))
    
    # Calculate precision and recall
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_positives, len(gold_set), len(pred_set)

def main(gold_annotation_dir, prediction_dir):
    """
    Main function to process note IDs and MIMIC-III notes into a sentence segmentation dataset.
    
    Args:
        gold_annotation_dir (str): Directory Path containing gold annotations.
        prediction_dir (str): Directory Path containing predictions.
    """
    # Load gold annotations and predictions
    gold_annotations = load_annotations(gold_annotation_dir)
    predictions = load_annotations(prediction_dir)
    
    print(f"Loaded {len(gold_annotations)} gold annotation files")
    print(f"Loaded {len(predictions)} prediction files")
    
    # Check if both sets have the same files
    gold_files = set(gold_annotations.keys())
    pred_files = set(predictions.keys())
    
    if gold_files != pred_files:
        error_msg = "Error: Gold annotations and predictions have different files!\n"
        error_msg += f"Files only in gold annotations: {gold_files - pred_files}\n"
        error_msg += f"Files only in predictions: {pred_files - gold_files}"
        raise ValueError(error_msg)
    
    # Calculate metrics for each file and for micro-average
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_files = 0
    
    # For micro-average
    total_true_positives = 0
    total_gold_boundaries = 0
    total_pred_boundaries = 0
    
    print("\nPer-file evaluation:")
    print("-" * 80)
    print(f"{'File ID':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 80)
    
    for file_id in gold_files:
        gold_boundaries = gold_annotations[file_id]
        pred_boundaries = predictions[file_id]
        
        precision, recall, f1, true_positives, gold_count, pred_count = calculate_metrics(gold_boundaries, pred_boundaries)
        
        print(f"{file_id:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        # For macro-average
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        num_files += 1
        
        # For micro-average
        total_true_positives += true_positives
        total_gold_boundaries += gold_count
        total_pred_boundaries += pred_count
    
    # Calculate and print macro-average metrics
    if num_files > 0:
        macro_precision = total_precision / num_files
        macro_recall = total_recall / num_files
        macro_f1 = total_f1 / num_files
        
        # Calculate micro-average metrics
        micro_precision = total_true_positives / total_pred_boundaries if total_pred_boundaries > 0 else 0
        micro_recall = total_true_positives / total_gold_boundaries if total_gold_boundaries > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        print("-" * 80)
        print("\nMacro-average metrics:")
        print(f"{'Metric':<15} {'Score':<10}")
        print("-" * 40)
        print(f"{'Precision':<15} {macro_precision:<10.4f}")
        print(f"{'Recall':<15} {macro_recall:<10.4f}")
        print(f"{'F1 Score':<15} {macro_f1:<10.4f}")
        
        print("\nMicro-average metrics:")
        print(f"{'Metric':<15} {'Score':<10}")
        print("-" * 40)
        print(f"{'Precision':<15} {micro_precision:<10.4f}")
        print(f"{'Recall':<15} {micro_recall:<10.4f}")
        print(f"{'F1 Score':<15} {micro_f1:<10.4f}")
    else:
        raise ValueError("No files found in the annotations!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate sentence segmentation results against gold annotations.')
    parser.add_argument('--gold_dir', required=True, help='Directory containing gold annotation CSV files')
    parser.add_argument('--pred_dir', required=True, help='Directory containing prediction CSV files')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.gold_dir):
        raise ValueError(f"Gold annotation directory does not exist: {args.gold_dir}")
    if not os.path.exists(args.pred_dir):
        raise ValueError(f"Prediction directory does not exist: {args.pred_dir}")
    
    main(args.gold_dir, args.pred_dir)
    
    