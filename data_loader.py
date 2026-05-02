"""Load and preprocess the Kaggle Fake and Real News dataset."""
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config


def load_kaggle_data():
    """
    Load dataset from data/ folder.
    Supports: (1) Fake.csv + True.csv  (2) news_merged.csv
    Returns DataFrame with columns: title, text, label (0=real, 1=fake).
    """
    if os.path.exists(config.MERGED_CSV):
        df = pd.read_csv(config.MERGED_CSV)
        if config.LABEL_COL not in df.columns:
            raise ValueError(f"Merged CSV must have column '{config.LABEL_COL}' (0=real, 1=fake).")
        return df

    if os.path.exists(config.FAKE_CSV) and os.path.exists(config.REAL_CSV):
        fake = pd.read_csv(config.FAKE_CSV)
        real = pd.read_csv(config.REAL_CSV)
        # Kaggle column names may be 'title', 'text', 'subject', 'date'
        title_col = "title" if "title" in fake.columns else fake.columns[0]
        text_col = "text" if "text" in fake.columns else fake.columns[1]
        fake = fake[[title_col, text_col]].copy()
        fake.columns = [config.TITLE_COL, config.TEXT_COL]
        fake[config.LABEL_COL] = 1
        real = real[[title_col, text_col]].copy()
        real.columns = [config.TITLE_COL, config.TEXT_COL]
        real[config.LABEL_COL] = 0
        return pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=config.RANDOM_STATE)

    raise FileNotFoundError(
        f"Place dataset in {config.DATA_DIR}: either 'Fake.csv' and 'True.csv', or 'news_merged.csv'"
    )


def clean_text(text):
    """Basic cleaning: lowercase, remove URLs and non-alphanumeric, collapse spaces."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_title_text(df):
    """Create a single 'content' column from title and text."""
    df = df.copy()
    title = df[config.TITLE_COL].fillna("")
    text = df[config.TEXT_COL].fillna("")
    df["content"] = (title + " " + text).apply(clean_text)
    return df


def get_train_val_test(df, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE):
    """Split into train, validation, test with stratification."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=config.RANDOM_STATE, stratify=df[config.LABEL_COL]
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train, test_size=val_ratio, random_state=config.RANDOM_STATE, stratify=train[config.LABEL_COL]
    )
    return train, val, test
