"""
Train an improved resolution-time model using the existing customer support dataset.

Run from the project root:
    python srC:/train_resolution.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(repo_root, "data", "raw", "customer_support_tickets.csv")
    models_dir = os.path.join(repo_root, "models")
    model_path = os.path.join(models_dir, "resolution_model.pkl")

    if not os.path.exists(data_path):
        # Fallback in case the dataset is stored directly under data/
        data_path = os.path.join(repo_root, "data", "customer_support_tickets.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "customer_support_tickets.csv was not found. "
            "Check the data/raw or data folder."
        )

    df = pd.read_csv(data_path)

    required = [
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Priority_Level",
        "Ticket_Channel",
        "Submission_Date",
        "Resolution_Time_Hours",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Text available when a new ticket is created.
    df["text"] = (
        df["Ticket_Subject"].fillna("").astype(str)
        + " "
        + df["Ticket_Description"].fillna("").astype(str)
    )

    # Use exact categorical values from the existing dataset.
    for col in ["Issue_Category", "Priority_Level", "Ticket_Channel"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Extract simple time features from submission date.
    dt = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df["submission_hour"] = dt.dt.hour.fillna(12)
    df["submission_dayofweek"] = dt.dt.dayofweek.fillna(0)
    df["submission_month"] = dt.dt.month.fillna(1)

    features = [
        "text",
        "Issue_Category",
        "Priority_Level",
        "Ticket_Channel",
        "submission_hour",
        "submission_dayofweek",
        "submission_month",
    ]

    X = df[features]
    y = df["Resolution_Time_Hours"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Issue_Category", "Priority_Level", "Ticket_Channel"],
            ),
            (
                "numeric",
                StandardScaler(with_mean=False),
                [
                    "submission_hour",
                    "submission_dayofweek",
                    "submission_month",
                ],
            ),
        ]
    )

    # Log-transform the target because resolution time is right-skewed.
    # This tends to reduce the impact of very long tickets on the model.
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=30.0)),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, 120)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, model_path)

    print("\nResolution model trained successfully.")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.3f}")
    print(f"\nSaved model to:\n{model_path}")


if __name__ == "__main__":
    main()
