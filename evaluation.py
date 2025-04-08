import pandas as pd
import os
import argparse
import numpy as np
import logging
import sys

# Add logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_annotations(annotation_file):
    """
    Load annotations from a single combined CSV file.
    
    Args:
        annotation_file (str): Path to the combined annotations CSV file
        
    Returns:
        dict: Dictionary mapping file IDs to their sentence boundaries
    """
    annotations = {}
    
    # Check if file exists
    if not os.path.exists(annotation_file):
        logging.error(f"Annotation file not found: {annotation_file}")
        sys.exit(1)
    
    # Read the combined CSV file
    df = pd.read_csv(annotation_file)
    
    # Group by file_id
    for file_id, group in df.groupby('file_id'):
        # Extract sentence boundaries for each file
        sentence_boundaries = []
        for _, row in group.iterrows():
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


def write_results_to_file(output_file, macro_metrics, micro_metrics):
    """Write evaluation results to output file in SMM4H format"""
    with open(output_file, 'w') as f:
        # Write micro-average results first
        f.write(f"micro_precision\t{micro_metrics['precision']:.4f}\n")
        f.write(f"micro_recall\t{micro_metrics['recall']:.4f}\n")
        f.write(f"micro_f1\t{micro_metrics['f1']:.4f}\n")
        
        # Write macro-average results
        f.write(f"macro_precision\t{macro_metrics['precision']:.4f}\n")
        f.write(f"macro_recall\t{macro_metrics['recall']:.4f}\n")
        f.write(f"macro_f1\t{macro_metrics['f1']:.4f}\n")

def main(gold_dir, pred_dir, output_dir):
    """
    Main function to process and evaluate sentence segmentation.
    """
    
    
    # Check if directories exist
    if not os.path.exists(pred_dir):
        logging.error(f"Prediction directory does not exist: {pred_dir}")
        sys.exit(1)

    # Load gold annotations and predictions
    predictions = load_annotations(pred_dir)
    gold_annotations = load_annotations(gold_dir)
    
    logging.info(f"Loaded {len(predictions)} prediction files")
    
    # Check if both sets have the same files
    pred_files = set(predictions.keys())
    gold_files = set(gold_annotations.keys())
    if gold_files != pred_files:
        error_msg = "Gold annotations and predictions have different files!"
        error_msg += f"\nFiles only in gold annotations: {gold_files - pred_files}"
        error_msg += f"\nFiles only in predictions: {pred_files - gold_files}"
        logging.error(error_msg)
        sys.exit(1)
    
    # Initialize counters
    total_precision = total_recall = total_f1 = 0
    total_true_positives = total_gold_boundaries = total_pred_boundaries = 0
    
    logging.info("Starting evaluation...")
    
    for file_id in gold_files:
        gold_boundaries = gold_annotations[file_id]
        pred_boundaries = predictions[file_id]
        
        precision, recall, f1, true_positives, gold_count, pred_count = calculate_metrics(
            gold_boundaries, pred_boundaries
        )
        
        # Update totals
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_true_positives += true_positives
        total_gold_boundaries += gold_count
        total_pred_boundaries += pred_count
    
    num_files = len(gold_files)
    if num_files == 0:
        logging.error("No files found in the annotations!")
        sys.exit(1)
    
    # Calculate macro and micro metrics
    macro_metrics = {
        'precision': total_precision / num_files,
        'recall': total_recall / num_files,
        'f1': total_f1 / num_files
    }
    
    micro_metrics = {
        'precision': total_true_positives / total_pred_boundaries if total_pred_boundaries > 0 else 0,
        'recall': total_true_positives / total_gold_boundaries if total_gold_boundaries > 0 else 0,
        'f1': 0
    }
    
    if (micro_metrics['precision'] + micro_metrics['recall']) > 0:
        micro_metrics['f1'] = 2 * (micro_metrics['precision'] * micro_metrics['recall']) / (
            micro_metrics['precision'] + micro_metrics['recall']
        )
    
    # Create output filename
    output_file = os.path.join(output_dir, 'scores.txt')
    
    # Write results to file
    write_results_to_file(output_file, macro_metrics, micro_metrics)
    logging.info(f"Results have been written to {output_file}")
    
    # Log the results
    logging.info("\nResults:")
    logging.info(f"Micro-average - P: {micro_metrics['precision']:.4f}, R: {micro_metrics['recall']:.4f}, F1: {micro_metrics['f1']:.4f}")
    logging.info(f"Macro-average - P: {macro_metrics['precision']:.4f}, R: {macro_metrics['recall']:.4f}, F1: {macro_metrics['f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate sentence segmentation results against gold annotations.')
    parser.add_argument('--gold_dir', required=True, help='Directory containing gold CSV files')
    parser.add_argument('--pred_dir', required=True, help='Directory containing prediction CSV files')
    parser.add_argument('--output_dir', required=True, help='Output file that contains evaluation scores.')
    args = parser.parse_args()
    

    
    main(args.gold_dir, args.pred_dir, args.output_dir)
    sys.exit(0)
    
    