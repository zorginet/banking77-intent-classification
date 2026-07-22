"""Module for text preprocessing and data preparation."""

import os
import random
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from wordcloud import STOPWORDS, WordCloud


def plot_top_ngrams(text_series, n=2, top_k=15, title="Top N-grams"):
    """Visualizes the most frequent n-grams in a text corpus.

    Args:
        text_series: Pandas Series with text data.
        n: Size of the n-gram (default 2 for bigrams).
        top_k: Number of top n-grams to display.
        title: Plot title.

    Displays:
        Horizontal bar chart of n-gram frequencies.
    """
    # Vectorize the text to get the frequency of each n-gram
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english")
    bag_of_words = vectorizer.fit_transform(text_series)
    sum_words = bag_of_words.sum(axis=0)
    # Create a list of (n-gram, frequency) pairs
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()
    ]
    # Sort and select the top-k most popular n-grams
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
    ngram_df = pd.DataFrame(words_freq, columns=["ngram", "count"])

    # Build a plot for visual comparison of frequencies
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=ngram_df["count"],
        y=ngram_df["ngram"],
        palette="mako",
        hue=ngram_df["count"],
        legend=False,
    )
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Ngram")
    plt.show()


# Extend the project's stopwords set: remove common words that interfere with intent detection
my_stopwords = set(STOPWORDS)
my_stopwords.update(
    [
        "want",
        "tell",
        "make",
        "now",
        "need",
        "know",
        "help",
        "hello",
        "hi",
        "one",
        "trying",
        "go",
        "think",
        "use",
        "tried",
        "issue",
        "let",
        "yet",
        "see",
    ]
)


# Display a word cloud for a quick visual overview of dominant terms in the corpus
def create_wordcloud(text, title):
    """Creates and displays a word cloud.

    Args:
        text (str): Text for generating the word cloud.
        title (str): Title for the image.

    Returns:
        None: The function displays a matplotlib figure.
    """
    wordcloud = WordCloud(
        stopwords=my_stopwords,
        background_color="white",
        colormap="mako",
        width=600,
        height=500,
        random_state=42,
    ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title, fontsize=14, pad=10)


# Ensure that NLTK has the necessary resources for tokenization and stopword filtering
def _ensure_nltk_data():
    """Checks and downloads the necessary NLTK resources.

    Ensures that 'stopwords' and the 'punkt' tokenizer are available. This prevents
    runtime errors when the module is imported in a new environment.

    Returns:
        None
    """
    # Download stopwords for removing unnecessary words
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # Download the tokenizer for splitting text into words
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# Let’s initialise the global tools for quick access in word processing
stemmer = SnowballStemmer(language="english")


def tokenize(text):
    """Cleans and stems the text.

    Args:
        text (str): Input text.

    Returns:
        list: A list of stemmed words without stopwords.
    """
    english_stopwords = stopwords.words("english")
    # Remove links, non-text characters, and convert to lowercase
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z'\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize the text
    tokens = word_tokenize(text)
    tokens_no_stop = [word for word in tokens if word.lower() not in english_stopwords]
    return [stemmer.stem(word) for word in tokens_no_stop]


def vectorize_text(train_text, val_text, test_text):
    """Converts text into TF-IDF numerical vectors.

    Args:
        train_text (list): Texts for training.
        val_text (list): Texts for validation.
        test_text (list): Texts for testing.

    Returns:
        tuple: TF-IDF matrices and the vectorizer.
    """
    _ensure_nltk_data()
    # Configure the TF-IDF vectorizer with n-grams to capture context
    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=tokenize,
        token_pattern=None,
        max_features=10000,
        ngram_range=(1, 3),
        min_df=3,
    )

    # Fit the vectorizer on the training data
    X_train_tfidf = vectorizer.fit_transform(train_text)
    # Apply the fitted parameters to the validation and test data
    X_val_tfidf = vectorizer.transform(val_text)
    X_test_tfidf = vectorizer.transform(test_text)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def set_seed(seed=42):
    """Sets random seeds for reproducibility.

    Args:
        seed (int): Seed value. Default is 42.
    """
    # Set seeds for all libraries to ensure reproducibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_tfidf_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128, seed=42
):
    """Creates DataLoaders for TF-IDF vectors.

    Args:
        X_train: Feature matrix for training.
        y_train: Labels for training.
        X_val: Feature matrix for validation.
        y_val: Labels for validation.
        X_test: Feature matrix for testing.
        y_test: Labels for testing.
        batch_size (int): Mini-batch size. Default is 128.
        seed (int): Seed for reproducibility. Default is 42.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    set_seed(seed)
    # Convert class labels to PyTorch tensors
    y_train_tensor = (
        torch.as_tensor(y_train.values, dtype=torch.long)
        if hasattr(y_train, "values")
        else torch.tensor(y_train, dtype=torch.long)
    )
    y_val_tensor = (
        torch.as_tensor(y_val.values, dtype=torch.long)
        if hasattr(y_val, "values")
        else torch.tensor(y_val, dtype=torch.long)
    )
    y_test_tensor = (
        torch.as_tensor(y_test.values, dtype=torch.long)
        if hasattr(y_test, "values")
        else torch.tensor(y_test, dtype=torch.long)
    )

    # Create a custom class for safe batch processing of sparse matrices
    class SparseDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            # Convert to a dense tensor only one row (batch) at a time, not the entire matrix
            x_row = (
                self.X[idx].toarray().squeeze(0)
                if hasattr(self.X, "toarray")
                else self.X[idx]
            )
            return torch.tensor(x_row, dtype=torch.float32), self.y[idx]

    # Create datasets without overloading RAM
    train_ds = SparseDataset(X_train, y_train_tensor)
    val_ds = SparseDataset(X_val, y_val_tensor)
    test_ds = SparseDataset(X_test, y_test_tensor)

    # Create DataLoaders for batch processing
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def prepare_st_dataloaders(
    model_st, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128, seed=42
):
    """Creates DataLoaders for Sentence Transformers representations.

    Args:
        model_st: Sentence Transformers model for encoding.
        X_train: Texts for training.
        y_train: Labels for training.
        X_val: Texts for validation.
        y_val: Labels for validation.
        X_test: Texts for testing.
        y_test: Labels for testing.
        batch_size (int): Mini-batch size. Default is 128.
        seed (int): Seed for reproducibility. Default is 42.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    set_seed(seed)
    # Encode texts into dense vector representations using the model
    X_train_st = model_st.encode(X_train.tolist(), convert_to_tensor=True)
    X_val_st = model_st.encode(X_val.tolist(), convert_to_tensor=True)
    X_test_st = model_st.encode(X_test.tolist(), convert_to_tensor=True)

    # Convert class labels to PyTorch tensors
    y_train_tensor = (
        torch.as_tensor(y_train.values, dtype=torch.long)
        if hasattr(y_train, "values")
        else torch.tensor(y_train, dtype=torch.long)
    )
    y_val_tensor = (
        torch.as_tensor(y_val.values, dtype=torch.long)
        if hasattr(y_val, "values")
        else torch.tensor(y_val, dtype=torch.long)
    )
    y_test_tensor = (
        torch.as_tensor(y_test.values, dtype=torch.long)
        if hasattr(y_test, "values")
        else torch.tensor(y_test, dtype=torch.long)
    )

    # Create datasets for Sentence Transformers
    train_ds_st = TensorDataset(X_train_st, y_train_tensor)
    val_ds_st = TensorDataset(X_val_st, y_val_tensor)
    test_ds_st = TensorDataset(X_test_st, y_test_tensor)

    # Create DataLoaders for batch processing
    train_dl_st = DataLoader(train_ds_st, batch_size=batch_size, shuffle=True)
    val_dl_st = DataLoader(val_ds_st, batch_size=batch_size, shuffle=False)
    test_dl_st = DataLoader(test_ds_st, batch_size=batch_size, shuffle=False)

    return train_dl_st, val_dl_st, test_dl_st


def prepare_bert_dataloaders(
    train_df,
    val_df,
    test_df,
    tokenizer,
    dataset_class,
    max_length=64,
    batch_size=32,
    seed=7,
):
    """Creates DataLoaders for BERT model.

    Args:
        train_df (pd.DataFrame): DataFrame for training data.
        val_df (pd.DataFrame): DataFrame for validation data.
        test_df (pd.DataFrame): DataFrame for testing data.
        tokenizer: BERT tokenizer.
        dataset_class: Dataset class for BERT.
        max_length (int): Maximum sequence length. Default is 64.
        batch_size (int): Mini-batch size. Default is 32.
        seed (int): Seed for reproducibility. Default is 42.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    set_seed(seed)
    # Tokenize and pad texts for BERT
    train_encoded = tokenizer(
        train_df["text"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    val_encoded = tokenizer(
        val_df["text"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    test_encoded = tokenizer(
        test_df["text"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(f"Train encoded keys: {train_encoded.keys()}")

    # Create datasets with tokenized texts and labels
    train_ds_bert = dataset_class(train_encoded, train_df["label"])
    val_ds_bert = dataset_class(val_encoded, val_df["label"])
    test_ds_bert = dataset_class(test_encoded, test_df["label"])

    # Create DataLoaders for batch processing
    train_dl_bert = torch.utils.data.DataLoader(
        train_ds_bert, batch_size=batch_size, shuffle=True
    )
    val_dl_bert = torch.utils.data.DataLoader(
        val_ds_bert, batch_size=batch_size, shuffle=False
    )
    test_dl_bert = torch.utils.data.DataLoader(
        test_ds_bert, batch_size=batch_size, shuffle=False
    )

    # Print information about the batch structure for debugging
    print(f"Batch sample keys: {next(iter(train_dl_bert)).keys()}")
    print(f"Input IDs shape: {next(iter(train_dl_bert))['input_ids'].shape}")
    print(f"Attention mask shape: {next(iter(train_dl_bert))['attention_mask'].shape}")
    print(f"Labels shape: {next(iter(train_dl_bert))['labels'].shape}")

    return train_dl_bert, val_dl_bert, test_dl_bert
