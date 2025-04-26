# Baseline sentence segmenter -- SliderSplit

## Description
This script [`SliderSplit.py`](SliderSplit.py) is used for sentence segmentation of MIMIC-III notes. It takes the clinical text as input and predict BIO tagging, where B indicates the Beginning of a sentence, I represents Inside of a sentence, and O denotes Outside of a sentence. More details of this model is in the paper [Automatic sentence segmentation of clinical record narratives in real-world data](https://aclanthology.org/2024.emnlp-main.1156/).

The model used in the script is at [dongfangxu/SentenceSegmenter-MIMIC](https://huggingface.co/dongfangxu/SentenceSegmenter-MIMIC).


#### Script Usage
The script contains a segmenter function to segment the clinical text 
```
sent_spans = segment_sentences(sample_text)
```
Example of the input and output of this function is shown in the main function.
