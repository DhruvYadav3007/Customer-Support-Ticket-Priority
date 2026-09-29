import os
import pandas as pd


def load_resolution_data():

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    path = os.path.join(
        repo_root,
        "data",
        "raw",
        "customer_support_tickets.csv"
    )

    if not os.path.exists(path):
        path = os.path.join(
            repo_root,
            "data",
            "customer_support_tickets.csv"
        )

    return pd.read_csv(path)


def get_resolution_range(priority, category):

    df = load_resolution_data()

    # First try exact Priority + Category combination
    subset = df[
        (df["Priority_Level"] == priority) &
        (df["Issue_Category"] == category)
    ]["Resolution_Time_Hours"].dropna()

    # If there aren't enough examples, fall back to priority only
    if len(subset) < 30:
        subset = df[
            df["Priority_Level"] == priority
        ]["Resolution_Time_Hours"].dropna()

    # Final fallback
    if len(subset) == 0:
        subset = df["Resolution_Time_Hours"].dropna()

    lower = subset.quantile(0.25)
    upper = subset.quantile(0.75)

    return round(lower, 1), round(upper, 1)


if __name__ == "__main__":

    print("\nResolution Range Tester")
    print("-----------------------")

    priority = input(
        "Priority (Critical/High/Medium/Low): "
    ).strip()

    category = input(
        "Category (Account/Billing/Fraud/General Inquiry/Technical): "
    ).strip()

    lower, upper = get_resolution_range(
        priority,
        category
    )

    print(
        f"\nExpected Resolution: {lower}–{upper} hours"
    )