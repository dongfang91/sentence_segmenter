import pandas as pd
import os
import argparse
import numpy as np

def main(note_ids_path, mimic_path, output_path):
    """
    Main function to process note IDs and MIMIC-III notes into a sentence segmentation dataset.
    
    Args:
        note_ids_path (str): Path to the text file containing note IDs.
        mimic_path (str): Directory path containing MIMIC-III CSV files.
        output_path (str): Output path for the processed corpus CSV file.
    """
    # Data loading
    arr_note_ids = []
    with open(note_ids_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                arr_note_ids.append(int(line))
    
    typesDict = {"ROW_ID": np.int32, "SUBJECT_ID": np.int32, "HADM_ID": str, "CHARTDATE": str, "CHARTTIME": str,
                     "STORETIME": str, "CATEGORY": str, "DESCRIPTION": str, "CGID": str, "ISERROR": str, "TEXT": str}

    df_mimic_note = pd.read_csv(
        os.path.join(mimic_path, 'NOTEEVENTS.csv.gz'),
        sep=',',
         dtype=typesDict
    )

    df_mimic_note.index = df_mimic_note["ROW_ID"].values
    df_data= df_mimic_note.loc[arr_note_ids, ["ROW_ID", "TEXT","CATEGORY","DESCRIPTION"]]

    # Save data
    output_folder = os.path.dirname(output_path)  # Get the folder path
    os.makedirs(output_folder, exist_ok=True)  
    df_data.to_csv(output_path, sep=',', index=False)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process and combine note IDs with MIMIC-III notes into a single dataset.")
    parser.add_argument(
        "-n",
        "--note_ids_path",
        type=str,
        required=True,
        help="Path to the text file containing note IDs"
    )
    parser.add_argument(
        "-m",
        "--mimic_path",
        type=str,
        required=True,
        help="Directory path containing the following CSV files from MIMIC-III v1.4: NOTEEVENTS.csv.gz, PRESCRIPTIONS.csv.gz and PATIENTS.csv.gz"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./corpus.csv",
        help="Output path for the processed corpus CSV file"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Run script
    main(
        note_ids_path=args.note_ids_path,
        mimic_path=args.mimic_path,
        output_path=args.output_path
    )