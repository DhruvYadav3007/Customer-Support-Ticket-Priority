import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    data_path = os.path.join(
        repo_root,
        "data",
        "raw",
        "customer_support_tickets.csv"
    )

    if not os.path.exists(data_path):
        data_path = os.path.join(
            repo_root,
            "data",
            "customer_support_tickets.csv"
        )

    df = pd.read_csv(data_path)

    # Only information that is realistically available
    # when a ticket is created.
    features = [
        "Priority_Level",
        "Issue_Category",
        "Ticket_Channel"
    ]

    target = "Resolution_Time_Hours"

    df = df.dropna(
        subset=features + [target]
    )

    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.clip(
        predictions,
        1,
        120
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== RESOLUTION MODEL V2 ==========")
    print(f"Rows: {len(df)}")
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R²:   {r2:.3f}")

    model_path = os.path.join(
        repo_root,
        "models",
        "resolution_model_v2.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nSaved:")
    print(model_path)


if __name__ == "__main__":
    main()