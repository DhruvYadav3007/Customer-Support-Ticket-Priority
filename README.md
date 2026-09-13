# 🎫 Customer Support Ticket Priority Prediction

An AI/ML-based application that automatically predicts the **priority of customer support tickets** based on the ticket description.

The project uses **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, and **Logistic Regression** to classify customer support requests. A **Streamlit web application** provides an interactive interface where users can enter a ticket description and receive a predicted priority.

---

## 📌 Project Overview

Customer support teams receive many tickets every day. Manually identifying which tickets require immediate attention can be time-consuming.

This project helps automate ticket prioritization by analyzing the text of a customer support ticket and predicting its priority using a trained Machine Learning model.

### Example

**Ticket Description**

```text
I am unable to access my account because the password reset link is not working.
I have tried several times but keep getting an error.
```

The trained model processes the description and predicts the appropriate ticket priority.

---

## ✨ Features

- Customer support ticket priority prediction
- Natural Language Processing
- TF-IDF text vectorization
- Logistic Regression classification
- Interactive Streamlit user interface
- Single ticket prediction
- Pre-trained ML model support
- Re-trainable machine learning pipeline
- Organized source code and model files

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- KaggleHub
- Jupyter Notebook

---

## 📂 Project Structure

```text
Customer-Support-Ticket-Priority/
│
├── data/
│   └── raw/
│       └── customer_support_tickets.csv
│
├── models/
│   ├── label_mapping.pkl
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── Notebooks/
│   └── datacollection.ipynb
│
├── src/
│   ├── datacollection.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> Generated processed datasets are not stored in GitHub because some TF-IDF files are very large. They can be recreated by running the preprocessing pipeline.

---

# 🚀 Getting Started

## 1. Clone the Repository

Open Terminal, PowerShell, Command Prompt, or the VS Code terminal.

```bash
git clone https://github.com/Please-use-me/Customer-Support-Ticket-Priority.git
```

Move into the project folder:

```bash
cd Customer-Support-Ticket-Priority
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### Windows

Create the environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Required Packages

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all project dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
kagglehub
pandas
scikit-learn
streamlit
```

---

# ▶️ Run the Application

If the trained model files are already available inside the `models/` folder, you do **not** need to train the model again.

Start the Streamlit application with:

```bash
streamlit run src/predict.py
```

After starting the application, Streamlit will display a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🖥️ Using the Application

1. Start the Streamlit application.
2. Open the displayed localhost URL.
3. Select **Single Ticket Prediction**.
4. Enter a customer support problem in the **Ticket Description** field.
5. Click **Predict**.
6. The ML model will predict the ticket priority.

### Example Input

```text
My payment was deducted twice for the same transaction and I have not received a refund.
Please resolve this issue as soon as possible.
```

---

# 🧠 Run the Complete ML Pipeline

If you want to rebuild the project from the beginning instead of using the existing trained model, follow these steps.

## Step 1 — Data Collection

```bash
python src/datacollection.py
```

This step collects/downloads the dataset required for the project.

---

## Step 2 — Data Preprocessing

```bash
python src/preprocessing.py
```

The preprocessing stage prepares the ticket text for Machine Learning.

Typical operations include:

- Data cleaning
- Handling ticket text
- Label processing
- Train/test splitting
- TF-IDF vectorization

Processed files are generated locally and are not committed to GitHub because of their large size.

---

## Step 3 — Train the Model

```bash
python src/train.py
```

The training process creates the Machine Learning model and saves the required files inside:

```text
models/
```

The important model files are:

```text
logistic_regression_model.pkl
tfidf_vectorizer.pkl
label_mapping.pkl
```

---

## Step 4 — Run the Prediction Application

After training is complete:

```bash
streamlit run src/predict.py
```

---

# 🔄 Complete Setup Commands

For macOS/Linux, the complete setup is:

```bash
git clone https://github.com/Please-use-me/Customer-Support-Ticket-Priority.git

cd Customer-Support-Ticket-Priority

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

streamlit run src/predict.py
```

If you want to retrain the model:

```bash
python src/datacollection.py

python src/preprocessing.py

python src/train.py

streamlit run src/predict.py
```

---

# 📊 Machine Learning Workflow

```text
Customer Support Dataset
          │
          ▼
     Data Cleaning
          │
          ▼
   Text Preprocessing
          │
          ▼
  TF-IDF Vectorization
          │
          ▼
  Train / Test Dataset
          │
          ▼
 Logistic Regression
          │
          ▼
      Trained Model
          │
          ▼
   Streamlit Web App
          │
          ▼
Ticket Priority Prediction
```

---

# 📦 Model Files

The application uses the following trained files:

| File | Purpose |
|---|---|
| `logistic_regression_model.pkl` | Trained classification model |
| `tfidf_vectorizer.pkl` | Converts ticket text into numerical features |
| `label_mapping.pkl` | Maps model output to ticket priority labels |

---

# ⚠️ Large Processed Files

TF-IDF transformed datasets can become very large.

For this reason, generated files such as:

```text
x_train_tfidf.csv
x_test_tfidf.csv
```

are excluded from GitHub.

They can be regenerated locally by running:

```bash
python src/preprocessing.py
```

This keeps the GitHub repository lightweight and easier to clone.

---

# ❓ Troubleshooting

## `streamlit: command not found`

Run:

```bash
pip install streamlit
```

or:

```bash
python -m streamlit run src/predict.py
```

---

## `ModuleNotFoundError`

Make sure the virtual environment is activated and run:

```bash
pip install -r requirements.txt
```

---

## Model File Not Found

If files inside the `models/` directory are missing, rebuild them:

```bash
python src/preprocessing.py
python src/train.py
```

Then run:

```bash
streamlit run src/predict.py
```

---

## Check Installed Packages

```bash
pip list
```

---

# 🔮 Future Improvements

Possible improvements include:

- Deep Learning based ticket classification
- BERT / Transformer models
- Automatic ticket routing
- Sentiment analysis
- Ticket category prediction
- Confidence scores
- REST API integration
- Database integration
- Cloud deployment
- Real-time customer support dashboards

---

# 👨‍💻 Author

**Suyash Biranje**

GitHub: [Please-use-me](https://github.com/Please-use-me)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and Machine Learning demonstration purposes.
