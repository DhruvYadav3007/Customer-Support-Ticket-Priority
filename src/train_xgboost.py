
import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from scipy.sparse import hstack

from xgboost import XGBClassifier


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
# MAIN
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

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

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

    # ========================================================
    # CLEAN TEXT
    # ========================================================

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

    # ========================================================
    # TEXT FEATURES
    # ========================================================

    print(
        "\nCreating TF-IDF features..."
    )

    subject_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        min_df=2,
        sublinear_tf=True
    )

    description_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=7500,
        min_df=2,
        sublinear_tf=True
    )

    # ========================================================
    # STRUCTURED FEATURES
    # ========================================================

    category_encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    channel_encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    indices = np.arange(
        len(df)
    )

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=df["Priority_Level"]
    )

    train_df = df.iloc[
        train_idx
    ]

    test_df = df.iloc[
        test_idx
    ]

    y_train = train_df[
        "Priority_Level"
    ]

    y_test = test_df[
        "Priority_Level"
    ]

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    # ========================================================
    # TF-IDF FIT ONLY ON TRAINING DATA
    # ========================================================

    X_subject_train = (
        subject_vectorizer
        .fit_transform(
            train_df["Ticket_Subject"]
        )
    )

    X_subject_test = (
        subject_vectorizer
        .transform(
            test_df["Ticket_Subject"]
        )
    )

    X_description_train = (
        description_vectorizer
        .fit_transform(
            train_df["Ticket_Description"]
        )
    )

    X_description_test = (
        description_vectorizer
        .transform(
            test_df["Ticket_Description"]
        )
    )

    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    X_category_train = (
        category_encoder
        .fit_transform(
            train_df[["Issue_Category"]]
        )
    )

    X_category_test = (
        category_encoder
        .transform(
            test_df[["Issue_Category"]]
        )
    )

    X_channel_train = (
        channel_encoder
        .fit_transform(
            train_df[["Ticket_Channel"]]
        )
    )

    X_channel_test = (
        channel_encoder
        .transform(
            test_df[["Ticket_Channel"]]
        )
    )

    # ========================================================
    # MANUAL TEXT STATISTICS
    # ========================================================

    def text_features(data):

        subject = data[
            "Ticket_Subject"
        ]

        description = data[
            "Ticket_Description"
        ]

        features = np.column_stack([

            subject.str.len(),

            description.str.len(),

            subject.str.split().str.len(),

            description.str.split().str.len(),

            description.str.count("!"),

            description.str.count(r"\?"),

            description.str.count(
                r"\b(error|failed|failure|urgent|critical|blocked|"
                r"cannot|unable|not working|issue|problem)\b"
            )

        ])

        return features.astype(float)

    numeric_train = text_features(
        train_df
    )

    numeric_test = text_features(
        test_df
    )

    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    X_train = hstack([

        X_subject_train,

        X_description_train,

        X_category_train,

        X_channel_train,

        numeric_train

    ]).tocsr()

    X_test = hstack([

        X_subject_test,

        X_description_test,

        X_category_test,

        X_channel_test,

        numeric_test

    ]).tocsr()

    print(
        "\nFinal feature shape:"
    )

    print(
        X_train.shape
    )

    # ========================================================
    # ENCODE TARGET
    # ========================================================

    labels = sorted(
        y_train.unique()
    )

    label_to_number = {
        label: index
        for index, label
        in enumerate(labels)
    }

    number_to_label = {
        index: label
        for label, index
        in label_to_number.items()
    }

    y_train_num = y_train.map(
        label_to_number
    )

    y_test_num = y_test.map(
        label_to_number
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=len(labels),

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=3,

        reg_lambda=1,

        eval_metric="mlogloss",

        tree_method="hist",

        random_state=42,

        n_jobs=-1

    )

    model.fit(
        X_train,
        y_train_num
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred_num = model.predict(
        X_test
    )

    y_pred = [
        number_to_label[
            int(value)
        ]
        for value in y_pred_num
    ]

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n========== XGBOOST PRIORITY MODEL =========="
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                y_pred
            ),
            4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )

    # ========================================================
    # SAVE MODEL + PREPROCESSORS
    # ========================================================

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        os.path.join(
            MODELS_DIR,
            "priority_xgboost_model.pkl"
        )
    )

    joblib.dump(
        subject_vectorizer,
        os.path.join(
            MODELS_DIR,
            "priority_subject_tfidf.pkl"
        )
    )

    joblib.dump(
        description_vectorizer,
        os.path.join(
            MODELS_DIR,
            "priority_description_tfidf.pkl"
        )
    )

    joblib.dump(
        category_encoder,
        os.path.join(
            MODELS_DIR,
            "priority_category_encoder.pkl"
        )
    )

    joblib.dump(
        channel_encoder,
        os.path.join(
            MODELS_DIR,
            "priority_channel_encoder.pkl"
        )
    )

    joblib.dump(
        {
            "label_to_number": label_to_number,
            "number_to_label": number_to_label
        },
        os.path.join(
            MODELS_DIR,
            "priority_label_mapping.pkl"
        )
    )

    print(
        "\n========== XGBOOST TRAINING COMPLETE =========="
    )

    print(
        "Model saved as:"
    )

    print(
        "models/priority_xgboost_model.pkl"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()

