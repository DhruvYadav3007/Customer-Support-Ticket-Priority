# Customer Support Ticket Prediction & Routing System

An AI-powered Customer Support Ticket Prediction and Routing System that automatically predicts ticket priority, estimates resolution time, routes tickets to the appropriate department, and provides batch-level ticket analysis.

## Author

**Dhruv Yadav**

## Project Overview

Customer support teams receive a large number of tickets through different channels. Manually determining the priority and routing every ticket can be time-consuming and inconsistent.

This project uses machine learning to automate the initial ticket-triage process.

The system can:

- Predict ticket priority
- Classify tickets into priority levels
- Estimate an expected resolution-time range
- Route tickets to the appropriate department
- Process individual tickets
- Process multiple tickets through CSV files
- Analyze batches of support tickets
- Display current tickets assigned to each department
- Export batch predictions as CSV
- Provide an included sample CSV for demonstration

## AI / Machine Learning

The priority prediction component uses an **XGBoost classifier**.

The model uses a combination of:

- Ticket subject TF-IDF features
- Ticket description TF-IDF features
- Issue category
- Ticket channel
- Additional text-based numerical features

The predicted priority levels are:

- Critical
- High
- Medium
- Low

## Ticket Routing

After predicting the priority, the ticket is routed according to its issue category.

| Issue Category | Department |
|---|---|
| Technical | Technical Support |
| Billing | Billing |
| Fraud | Fraud & Risk |
| Account | Account Support |
| General Inquiry | General Support |

Each department maintains its own ticket queue.

## Batch Prediction

The application supports CSV-based batch processing.

Users can either:

1. Upload their own CSV file
2. Click **Use Sample CSV** to process the included demonstration dataset

The batch analysis displays:

- Total tickets
- Number of departments
- Critical ticket count
- Priority distribution
- Department distribution
- Priority by department
- Complete prediction results

The resulting predictions can also be downloaded as a CSV file.

## Sample Dataset

A sample CSV is included for demonstration:

```text
sample_data/sample_tickets.csv
```

This allows the application to be tested without preparing a CSV manually.

## Project Structure

```text
Customer-Support-Ticket-Priority/
│
├── data/
│   └── raw/
│       └── customer_support_tickets.csv
│
├── models/
│   ├── priority_xgboost_model.pkl
│   ├── priority_subject_tfidf.pkl
│   ├── priority_description_tfidf.pkl
│   ├── priority_category_encoder.pkl
│   ├── priority_channel_encoder.pkl
│   ├── priority_label_mapping.pkl
│   └── resolution_model.pkl
│
├── sample_data/
│   └── sample_tickets.csv
│
├── src/
│   ├── datacollection.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_xgboost.py
│   ├── train_catboost.py
│   ├── train_resolution.py
│   ├── train_resolution_v2.py
│   ├── resolution_range.py
│   └── predict.py
│
├── requirements.txt
├── run.bat
└── README.md
```

## Running the Application

### Windows

Activate the virtual environment:

```bash
venv\Scripts\activate
```

Start the application:

```bash
python -m streamlit run src/predict.py
```

Alternatively, double-click:

```text
run.bat
```

## Application Workflow

```text
Support Ticket
      |
      v
Text & Ticket Features
      |
      v
TF-IDF + Encoded Features
      |
      v
XGBoost Priority Classifier
      |
      +-- Critical
      +-- High
      +-- Medium
      +-- Low
      |
      v
Department Routing
      |
      +-- Technical Support
      +-- Billing
      +-- Fraud & Risk
      +-- Account Support
      +-- General Support
      |
      v
Department Queue
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- SciPy
- Streamlit
- TF-IDF
- Machine Learning Classification

## Main Features

### Single Ticket Prediction

Enter:

- Ticket subject
- Ticket description
- Issue category
- Ticket channel

The system predicts the ticket priority and displays the expected resolution range.

### Ticket Routing

A predicted ticket can be routed to its appropriate department and stored in that department's queue.

### Department Queues

The application displays the current tickets in each department, including queue sequence numbers.

### Batch Processing

Multiple tickets can be processed at once using a CSV file.

### Batch Analysis

The application provides visual and tabular analysis of the processed tickets.

## Author

**Dhruv Yadav**

GitHub:  
https://github.com/DhruvYadav3007


