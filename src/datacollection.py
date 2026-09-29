from os import path

import kagglehub
import pandas as pd
import os

def collection():
    
    path = kagglehub.dataset_download("ajverse/customer-support-tickets-crm-dataset")

    print("Path to dataset files:", path)
    df = pd.read_csv(str(path) + "/customer_support_tickets.csv")
    df.head()

    os.makedirs("C:/MCA/ticket/Customer-Support-Ticket-Priority/data/raw", exist_ok=True)
    df.to_csv("C:/MCA/ticket/Customer-Support-Ticket-Priority/data/raw/customer_support_tickets.csv", index=False)
if __name__ == "__main__":
    collection()



