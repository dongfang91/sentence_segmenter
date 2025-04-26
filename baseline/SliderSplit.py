import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def extract_token_index(note_raw):
    """
    Split the raw clinical text into tokens and track their character indices.
    
    Args:
        note_raw (str): Input text to be tokenized
        
    Returns:
        tuple: (tokens_list, tokens_ori_list, index_list)
            - tokens_list: list of processed tokens
            - tokens_ori_list: list of original tokens
            - index_list: list of starting character indices for each token
    """
    note_raw_newline = re.sub('\n', ' <newline> ', note_raw)
    note_raw_deidentified = re.sub('\[\*\*(.*?)\*\*]', ' <deidentified> ', note_raw_newline)

    tokens_list = []
    index_list = []
    tokens_ori_list = []
    text_start = 0
    start = 0
    
    for token in note_raw_deidentified.split():
        if token == "<newline>":
            idx_start = note_raw.index('\n', start)
            idx_end = idx_start+1
            start = idx_end
            token_raw = '\n'
        elif token == '<deidentified>':
            match = re.search('\[\*\*(.*?)\*\*]', note_raw[start:])
            if match:
                sub_string_start,sub_string_end = match.start(), match.end()
                idx_start = sub_string_start + start
                idx_end = sub_string_end + start
                token_raw = match.group()
            else:
                print("match not found...". note_raw[start:start+10])
        else:
            idx_start = note_raw.index(token, start)
            idx_end = idx_start+len(token)
            start = idx_end
            token_raw = token
        
        tokens_list.append(token)
        tokens_ori_list.append(token_raw)
        index_list.append(text_start+idx_start)# print(token, note_raw[idx_start: idx_end])

    return tokens_list, tokens_ori_list,index_list

def get_tokens_idx(tokenizer, tokens_list, start_token_index, sliding_window_limit):
    """
    Calculate the end token index for a sliding window.
    
    Args:
        tokenizer: HuggingFace tokenizer
        tokens_list (list): List of tokens
        start_token_index (int): Starting token index
        sliding_window_limit (int): Maximum window size
        
    Returns:
        int: End token index
    """
    # tokens_window = tokens_list[start_token_index:start_token_index+sliding_window_limit]
    tokens_window =  tokens_list[start_token_index:]
    tokens_sent_input = tokenizer(tokens_window, is_split_into_words=True, truncation=False)
    sub_token = tokens_sent_input.tokens()[1:-1]
    word_id = tokens_sent_input.word_ids()[1:-1]
    
    if len(sub_token) >= sliding_window_limit:
        end_token = word_id[sliding_window_limit-1]-1
    else:
        end_token = len(tokens_window)
    
    return end_token + start_token_index

def find_index_subtokens(word_ids):
    """Find indices of first subtokens for each word."""
    indices = []
    previous_word_idx = None
    for index, word_idx in enumerate(word_ids):
        if word_idx != previous_word_idx:
            indices.append(index)
        previous_word_idx = word_idx
    return indices

def get_sent_end_index(tokenizer, model, tokens_window, tokens_window_ori):
    """
    Find the end index of the current sentence in the token window.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        tokens_window (list): Window of tokens to process
        tokens_window_ori (list): Original tokens
        
    Returns:
        tuple: (current_token_end_index, next_sent_index, tokens_prediction, tokens_prediction_tag)
    """
    inputs_token = tokenizer(tokens_window, is_split_into_words=True, truncation=True)
    inputs_token_subtokens_id = inputs_token.word_ids()[1:-1]
    subtoken_word_id = find_index_subtokens(inputs_token_subtokens_id)
    
    inputs = tokenizer(tokens_window, is_split_into_words=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_subtoken_class = [model.config.id2label[t.item()] for t in predictions[0]][1:-1]
    
    predicted_token_class = [predicted_subtoken_class[item] for item in subtoken_word_id]
    
    if 'B-Sent' in predicted_token_class[1:]:
        next_sent_index = predicted_token_class[1:].index('B-Sent') + 1
        back_track = 1
        while predicted_token_class[next_sent_index-back_track] == "O" or tokens_window[next_sent_index-back_track] == "<newline>":
            back_track += 1
        current_token_end_index = next_sent_index - back_track
    else:
        next_sent_index = len(predicted_token_class)
        current_token_end_index = len(predicted_token_class) - 1
    
    tokens_prediction = tokens_window[:next_sent_index]
    tokens_prediction_tag = predicted_token_class[:current_token_end_index+1] + ["O"] * (next_sent_index - 1 - current_token_end_index)
    
    return current_token_end_index, next_sent_index, tokens_prediction, tokens_prediction_tag

def get_token_length(token):
    """Get the length of a token, handling special characters <newline>."""
    return 1 if token == '<newline>' else len(token)

def segment_sentences(text, model_name="dongfangxu/SentenceSegmenter-MIMIC"):
    """
    Segment a text into sentences using a BIO tagging model.
    
    Args:
        text (str): Input text to segment
        model_name (str): the path of the sentence boundary BIO tagging model
        
    Returns:
        list: List of tuples containing (start_index, end_index) for each sentence
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Process text
    text = text.rstrip()
    tokens, tokens_ori, indices = extract_token_index(text)
    sent_spans = []
    start_token = 0
    sliding_window_limit = 500
    
    while start_token < len(tokens):
        end_token_window_id = get_tokens_idx(tokenizer, tokens, start_token, sliding_window_limit)
        current_end_token_id, next_start_token_id, _, _ = get_sent_end_index(
            tokenizer, 
            model, 
            tokens[start_token:end_token_window_id],
            tokens_ori[start_token:end_token_window_id]
        )
        
        next_start_token_id += start_token
        current_end_token_id += start_token
        
        start_token_index = indices[start_token]
        end_token_index = indices[current_end_token_id] + get_token_length(tokens_ori[current_end_token_id])
        
        sent_spans.append((start_token_index, end_token_index))
        start_token = next_start_token_id
    
    return sent_spans

def main():
    # Example usage
    sample_text = """Patient was admitted on 2/5/2023 Chief complaint: fever and cough
Past Medical History:
Hypertension, Diabetes
Patient reports worsening symptoms over the past week."""
    
    print("Original text:")
    print(sample_text)
    print("\nSegmented sentences:")
    
    sent_spans = segment_sentences(sample_text)
    for span in sent_spans:
        print("-" * 40)
        print(sample_text[span[0]:span[1]])
    print("-" * 40)

if __name__ == "__main__":
    main()


