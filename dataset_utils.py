import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Text cleaning / tokenization
def simple_tokenize(text):
    # Simple lowercase tokenizer for the BiLSTM pipeline.
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def truncate_tokens(tokens, max_length, strategy="head-only"):
    # Truncate a token list according to the project ablation settings.
    # head-only : keep the first max_length tokens
    # head+tail : keep the first half and last half

    if len(tokens) <= max_length:
        return tokens

    if strategy == "head-only":
        return tokens[:max_length]

    if strategy == "head+tail":
        head_len = max_length // 2
        tail_len = max_length - head_len
        return tokens[:head_len] + tokens[-tail_len:]

    raise ValueError(f"Unknown truncation strategy: {strategy}")



def add_length_columns(df):

    # Add both word-count and simple-token-count columns.

    # word-count length (intuitive)
    # token-count length (model processing)


    df = df.copy()

    token_lists = df["text"].apply(simple_tokenize)
    df["review_word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["review_token_count"] = token_lists.apply(len)

    def assign_bin(token_count):
        if token_count <= 128:
            return "short"
        elif token_count <= 512:
            return "medium"
        else:
            return "long"

    df["length_bin"] = df["review_token_count"].apply(assign_bin)
    return df


def load_imdb_data(validation_size=5000, random_state=42):

    # Load IMDB from Hugging Face
    # The train/validation split is stratified by sentiment label & review length bin

    dataset = load_dataset("imdb")

    train_full_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_full_df = add_length_columns(train_full_df)
    test_df = add_length_columns(test_df)

    train_full_df["stratify_key"] = (
        train_full_df["label"].astype(str) + "_" + train_full_df["length_bin"].astype(str)
    )

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_full_df["stratify_key"]
    )

    train_df = train_df.drop(columns=["stratify_key"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["stratify_key"]).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df

# vocab helpers
def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text_for_bilstm(text, vocab, max_length, truncation_strategy="head-only"):
    # Convert raw text to a padded list of token ids for the BiLSTM.

    tokens = simple_tokenize(text)
    tokens = truncate_tokens(tokens, max_length, truncation_strategy)

    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(token_ids) < max_length:
        token_ids = token_ids + [vocab["<PAD>"]] * (max_length - len(token_ids))

    return token_ids[:max_length]
