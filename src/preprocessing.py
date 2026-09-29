
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pandas as pd
import joblib
import os


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "customer_support_tickets.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed_data"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Keep letters, spaces and apostrophes
    text = re.sub(
        r"[^a-zA-Z\s']",
        "",
        text
    )

    return text.strip()


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data():

    print(
        "Loading dataset:"
    )

    print(
        DATA_PATH
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nDataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "Ticket_Description",
        "Ticket_Subject",
        "Issue_Category",
        "Priority_Level"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Combine useful ticket information
    # --------------------------------------------------------

    df["model_text"] = (
        "category "
        + df["Issue_Category"]
        .fillna("")
        .astype(str)
        + " subject "
        + df["Ticket_Subject"]
        .fillna("")
        .astype(str)
        + " description "
        + df["Ticket_Description"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    x = df[
        "model_text"
    ]

    y = df[
        "Priority_Level"
    ]

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    x_train = x_train.apply(
        clean_text
    )

    x_test = x_test.apply(
        clean_text
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print(
        "\nCreating TF-IDF features..."
    )

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )

    x_train_tfidf = (
        tfidf_vectorizer
        .fit_transform(x_train)
    )

    x_test_tfidf = (
        tfidf_vectorizer
        .transform(x_test)
    )

    print(
        "TF-IDF shape:",
        x_train_tfidf.shape
    )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    label_mapping = {
        label: idx
        for idx, label
        in enumerate(
            sorted(
                y_train.unique()
            )
        )
    }

    y_train_num = y_train.map(
        label_mapping
    )

    y_test_num = y_test.map(
        label_mapping
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save vectorizer
    # --------------------------------------------------------

    joblib.dump(
        tfidf_vectorizer,
        os.path.join(
            MODELS_DIR,
            "tfidf_vectorizer.pkl"
        )
    )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    joblib.dump(
        label_mapping,
        os.path.join(
            MODELS_DIR,
            "label_mapping.pkl"
        )
    )

    # --------------------------------------------------------
    # Save TF-IDF training data
    # --------------------------------------------------------

    x_train_tfidf_df = (
        pd.DataFrame
        .sparse
        .from_spmatrix(
            x_train_tfidf,
            columns=
            tfidf_vectorizer
            .get_feature_names_out()
        )
    )

    x_train_tfidf_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "x_train_tfidf.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save TF-IDF test data
    # --------------------------------------------------------

    x_test_tfidf_df = (
        pd.DataFrame
        .sparse
        .from_spmatrix(
            x_test_tfidf,
            columns=
            tfidf_vectorizer
            .get_feature_names_out()
        )
    )

    x_test_tfidf_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "x_test_tfidf.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    y_train_num.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "y_train_num.csv"
        ),
        index=False
    )

    y_test_num.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "y_test_num.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n========== PREPROCESSING COMPLETE =========="
    )

    print(
        f"Training rows: {len(x_train)}"
    )

    print(
        f"Test rows: {len(x_test)}"
    )

    print(
        f"TF-IDF features: {x_train_tfidf.shape[1]}"
    )

    print(
        "\nLabel mapping:"
    )

    print(
        label_mapping
    )

    print(
        "\nSaved to:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        MODELS_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    preprocess_data()
