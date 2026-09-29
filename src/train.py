
import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s']",
        "",
        text
    )

    return text.strip()


# ============================================================
# TRAIN
# ============================================================

def train():

    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Ticket_Channel",
        "Priority_Level"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean text fields
    # --------------------------------------------------------

    df["Ticket_Subject"] = (
        df["Ticket_Subject"]
        .fillna("")
        .apply(clean_text)
    )

    df["Ticket_Description"] = (
        df["Ticket_Description"]
        .fillna("")
        .apply(clean_text)
    )

    df["Issue_Category"] = (
        df["Issue_Category"]
        .fillna("Unknown")
        .astype(str)
    )

    df["Ticket_Channel"] = (
        df["Ticket_Channel"]
        .fillna("Unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df[
        [
            "Ticket_Subject",
            "Ticket_Description",
            "Issue_Category",
            "Ticket_Channel"
        ]
    ]

    y = df[
        "Priority_Level"
    ]

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows: {len(X_test)}"
    )

    # ========================================================
    # FEATURE TRANSFORMER
    # ========================================================

    print(
        "\nBuilding feature pipeline..."
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "subject_tfidf",

                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=2500,
                    min_df=2,
                    sublinear_tf=True
                ),

                "Ticket_Subject"
            ),

            (
                "description_tfidf",

                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=7500,
                    min_df=2,
                    sublinear_tf=True
                ),

                "Ticket_Description"
            ),

            (
                "category",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                ["Issue_Category"]
            ),

            (
                "channel",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                ["Ticket_Channel"]
            )
        ]
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = Pipeline(
    steps=[
        ("features", preprocessor),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced"
            )
        )
    ]
)

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining combined-feature model..."
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    y_pred = model.predict(
        X_test
    )

    print(
        "\n========== COMBINED PRIORITY MODEL =========="
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODELS_DIR,
        "priority_pipeline.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\n========== TRAINING COMPLETE =========="
    )

    print(
        "Saved:"
    )

    print(
        model_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()
