from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import pandas as pd
import joblib
import os

def preprocess_data(path):
    df=pd.read_csv(path)

    x = df["Ticket_Description"].fillna("")
    y = df["Priority_Level"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    def clean_reviews(i):
        i = str(i).lower()
        i = re.sub(r"<[^>]+>", "", i)
        i = re.sub(r"https?://\S+|www\.\S+", "", i)
        i = re.sub(r"[^a-zA-Z\s']", "", i)
        return i.strip()

    x_train = x_train.apply(clean_reviews)
    x_test = x_test.apply(clean_reviews)

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    label_mapping = {label: idx for idx, label in enumerate(sorted(y_train.unique()))}
    y_train_num = y_train.map(label_mapping)
    y_test_num = y_test.map(label_mapping)

    os.makedirs("/Users/suyashbiranje/Desktop/Customer_Support/models", exist_ok=True)
    os.makedirs("/Users/suyashbiranje/Desktop/Customer_Support/processed_data", exist_ok=True)
    joblib.dump(tfidf_vectorizer, "/Users/suyashbiranje/Desktop/Customer_Support/models/tfidf_vectorizer.pkl")
    joblib.dump(label_mapping, "/Users/suyashbiranje/Desktop/Customer_Support/models/label_mapping.pkl")
    x_train_tfidf_df = pd.DataFrame.sparse.from_spmatrix(x_train_tfidf,columns=tfidf_vectorizer.get_feature_names_out())
    x_train_tfidf_df.to_csv("/Users/suyashbiranje/Desktop/Customer_Support/processed_data/x_train_tfidf.csv", index=False)
    x_test_tfidf_df = pd.DataFrame.sparse.from_spmatrix(x_test_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
    x_test_tfidf_df.to_csv("/Users/suyashbiranje/Desktop/Customer_Support/processed_data/x_test_tfidf.csv", index=False)
    y_train_num.to_csv("/Users/suyashbiranje/Desktop/Customer_Support/processed_data/y_train_num.csv", index=False)
    y_test_num.to_csv("/Users/suyashbiranje/Desktop/Customer_Support/processed_data/y_test_num.csv", index=False)

if __name__ == "__main__":
    path="/Users/suyashbiranje/Desktop/Customer_Support/data/raw/customer_support_tickets.csv"
    preprocess_data(path)