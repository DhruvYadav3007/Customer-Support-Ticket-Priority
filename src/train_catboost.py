
import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from catboost import CatBoostClassifier


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

    return text.strip()


# ============================================================
# TEXT STATISTICS
# ============================================================

def add_text_features(df):

    df["Subject_Length"] = (
        df["Ticket_Subject"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["Description_Length"] = (
        df["Ticket_Description"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["Subject_Word_Count"] = (
        df["Ticket_Subject"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["Description_Word_Count"] = (
        df["Ticket_Description"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["Exclamation_Count"] = (
        df["Ticket_Description"]
        .fillna("")
        .astype(str)
        .str.count("!")
    )

    df["Question_Count"] = (
        df["Ticket_Description"]
        .fillna("")
        .astype(str)
        .str.count(r"\?")
    )

    # Words that may indicate urgency
    urgency_pattern = (
        r"\b("
        r"urgent|urgently|critical|emergency|"
        r"immediately|asap|blocked|failure|failed|"
        r"cannot|unable|outage|down|security|fraud"
        r")\b"
    )

    df["Urgency_Word_Count"] = (
        df["Ticket_Description"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.count(urgency_pattern)
    )

    return df


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
        "Submission_Date",
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
    # CLEAN DATA
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
    # DATE FEATURES
    # ========================================================

    df["Submission_Date"] = pd.to_datetime(
        df["Submission_Date"],
        errors="coerce"
    )

    df["Submission_Year"] = (
        df["Submission_Date"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    df["Submission_Month"] = (
        df["Submission_Date"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    df["Submission_Day"] = (
        df["Submission_Date"]
        .dt.day
        .fillna(0)
        .astype(int)
    )

    df["Submission_DayOfWeek"] = (
        df["Submission_Date"]
        .dt.dayofweek
        .fillna(0)
        .astype(int)
    )

    df["Submission_Hour"] = (
        df["Submission_Date"]
        .dt.hour
        .fillna(0)
        .astype(int)
    )

    # ========================================================
    # TEXT FEATURES
    # ========================================================

    df = add_text_features(df)

    # ========================================================
    # FEATURES
    # ========================================================

    feature_columns = [

        "Ticket_Subject",
        "Ticket_Description",

        "Issue_Category",
        "Ticket_Channel",

        "Submission_Year",
        "Submission_Month",
        "Submission_Day",
        "Submission_DayOfWeek",
        "Submission_Hour",

        "Subject_Length",
        "Description_Length",

        "Subject_Word_Count",
        "Description_Word_Count",

        "Exclamation_Count",
        "Question_Count",

        "Urgency_Word_Count"
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        "Priority_Level"
    ].copy()

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = [
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Ticket_Channel"
    ]

    # CatBoost needs categorical values as strings
    for column in categorical_columns:

        X[column] = (
            X[column]
            .fillna("")
            .astype(str)
        )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

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
    # CATBOOST
    # ========================================================

    print(
        "\nTraining CatBoost..."
    )

    categorical_indices = [
        X.columns.get_loc(column)
        for column in categorical_columns
    ]

    model = CatBoostClassifier(

        loss_function="MultiClass",

        iterations=500,

        depth=7,

        learning_rate=0.05,

        l2_leaf_reg=5,

        random_seed=42,

        eval_metric="MultiClass",

        verbose=100,

        thread_count=-1
    )

    model.fit(

        X_train,

        y_train,

        cat_features=categorical_indices,

        eval_set=(X_test, y_test),

        early_stopping_rounds=50
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test
    )

    # CatBoost returns shape (n, 1)
    y_pred = y_pred.flatten()

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n========== CATBOOST PRIORITY MODEL =========="
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = sorted(
        y.unique()
    )

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
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODELS_DIR,
        "priority_catboost_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    # Save feature information
    metadata = {

        "feature_columns": feature_columns,

        "categorical_columns": categorical_columns,

        "labels": labels
    }

    joblib.dump(

        metadata,

        os.path.join(
            MODELS_DIR,
            "priority_catboost_metadata.pkl"
        )
    )

    print(
        "\n========== CATBOOST TRAINING COMPLETE =========="
    )

    print(
        "Model saved as:"
    )

    print(
        model_path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()
