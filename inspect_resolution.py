import pandas as pd
import os

path = "data/raw/customer_support_tickets.csv"

if not os.path.exists(path):
    path = "data/customer_support_tickets.csv"

df = pd.read_csv(path)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== RESOLUTION STATISTICS ==========")
print(df["Resolution_Time_Hours"].describe())

print("\n========== BY PRIORITY ==========")
print(
    df.groupby("Priority_Level")["Resolution_Time_Hours"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print("\n========== BY CATEGORY ==========")
print(
    df.groupby("Issue_Category")["Resolution_Time_Hours"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print("\n========== BY CHANNEL ==========")
print(
    df.groupby("Ticket_Channel")["Resolution_Time_Hours"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)