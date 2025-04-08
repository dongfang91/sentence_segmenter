# BioCreative IX @ IJCAI Track 2: Sentence segmentation of real-life clinical notes

We invite researchers and practitioners to participate in the **Track 2: Sentence segmentation of real-life clinical notes** at the [**BioCreative IX @ IJCAI**](https://healthlanguageprocessing.org/clinical-sentence-segmentation/). This task focuses on developing automatic systems that can accurately segment clinical notes into sentences. This task requires participants to identify spans of each sentence.

## Task Details

Sentence segmentation is a fundamental linguistic task widely used as a pre-processing step in many NLP applications. While modern LLMs and sparse attention mechanisms in transformer networks have reduced the necessity of sentence-level inputs in some NLP tasks, many models are still designed and tested for shorter sequences. The need for sentence segmentation is particularly pronounced in clinical notes, as most clinical NLP tasks depend on this information for annotation and model training. In 2024, we introduced a  **[baseline system](https://aclanthology.org/2024.emnlp-main.1156.pdf)** for sentence segmentation that outperformed several standard tools, such as Stanza, SpaCy, and Trankit. To foster further innovation in sentence segmentation techniques, we organized a shared task as part of the BioCreative 9 workshop.

## Registration and Resources

To join this shared task, please register for the **Track 2: Sentence segmentation of real-life clinical notes** through the [BioCreative IX Shared Task Registration Form](https://forms.gle/xbQp158cn5pgJ1oj9). Upon registration, participants will gain access to the full dataset and annotation guideline.

## Requirements

The following requirements apply to both data preparation and evaluation scripts:
- Python 3.6 or higher
- pandas library

Additional requirement for evaluation:
- numpy library

## Corpus and Data Preparation

This shared task utilizes a corpus of clinical notes derived from the MIMIC-III Database.

### Accessing MIMIC-III Notes

Participants are required to complete necessary training and sign a data usage agreement to access the [MIMIC-III Clinical Database (v1.4)](https://physionet.org/content/mimiciii/1.4/). After gaining access and downloading the files, participants must run the [`collect_notes_from_mimic.py`](collect_notes_from_mimic.py) script to retrieve clinical notes using the provided ROW_ID.

#### Script Usage
The script requires three command-line arguments:
- `--note_ids_path`: The file path to the text file containing the note IDs.
- `--mimic_path`: The directory path containing the MIMIC-III v1.4 CSV files (`NOTEEVENTS.csv.gz`, `PRESCRIPTIONS.csv.gz` and `PATIENTS.csv.gz`).
- `--output_path`: The file path where the processed corpus CSV will be saved.

##### Command Syntax
```
python collect_notes_from_mimic.py --note_ids_path data/biocreative/sample/sample_row_id.txt  --mimic_path ./mimic-iii/1.4 --output_path data/biocreative/sample/sample_corpus.csv
```

### Annotations Format

Ground truth annotations are provided in CSV format. A sample of the training set annotations is available at [`data/biocreative/sample`](data/biocreative/sample). Each annotation file is named using the ROW_ID of clinical notes from the MIMIC-III and includes information on sentence boundaries and types. There are two types of text chunks:
- Sentence
- Unstructured

The distinction between these types is detailed in our annotation guidelines, which will be provided upon registration. Registered participants will have access to the full dataset and annotation guidelines.

## Evaluation

The evaluation metrics for this task is adapted from the 2018 Universal Dependency Parsing Shared task, where we compared the sentence boundaries from the gold annotation. The `evaluation.py` is our evaluation script.

### Script Usage
The script requires two command-line arguments:
- `--gold_dir`: Directory containing gold annotation CSV files
- `--pred_dir`: Directory containing prediction CSV files
- `--output_dir`: Directory saving evaluation scores

#### Command Syntax
```
python evaluation.py --gold_dir data/biocreative/sample/annotation --pred_dir path/to/your/predictions --output_dir path/to/your/scores
```

The script outputs a formatted table showing:
- Individual file performance
- Macro-average metrics (precision, recall, F1)
- Micro-average metrics (precision, recall, F1)
