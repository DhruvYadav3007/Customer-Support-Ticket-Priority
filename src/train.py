from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import pandas as pd
import joblib
import os


def preprocess_data(path):
    x_train=pd.read_csv(r'/Users/suyashbiranje/Desktop/Customer_Support/data/processed_data/x_train_tfidf.csv')
    y_train=pd.read_csv(r'/Users/suyashbiranje/Desktop/Customer_Support/data/processed_data/y_train_num.csv')
    x_test=pd.read_csv(r'/Users/suyashbiranje/Desktop/Customer_Support/data/processed_data/x_test_tfidf.csv')
    y_test=pd.read_csv(r'/Users/suyashbiranje/Desktop/Customer_Support/data/processed_data/y_test_num.csv')


    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)     
    print(classification_report(y_test.values.ravel(), y_pred))
    
    os.makedirs("/Users/suyashbiranje/Desktop/Customer_Support/models", exist_ok   
=True)
    joblib.dump(model, "/Users/suyashbiranje/Desktop/Customer_Support/models/logistic_regression_model.pkl")


if __name__ == "__main__":
    path="/Users/suyashbiranje/Desktop/Customer_Support/"
    preprocess_data(path)

