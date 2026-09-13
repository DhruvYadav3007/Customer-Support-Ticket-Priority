("""
Streamlit app for predicting ticket priority using trained model.

Run: `streamlit run src/predict.py`
""")

import os
import streamlit as st
import pandas as pd
import joblib
from typing import Optional


def get_paths():
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	models_dir = os.path.join(repo_root, "models")
	return {
		"tfidf": os.path.join(models_dir, "tfidf_vectorizer.pkl"),
		"model": os.path.join(models_dir, "logistic_regression_model.pkl"),
		"labels": os.path.join(models_dir, "label_mapping.pkl"),
	}


@st.cache_resource
def load_artifacts(tfidf_path: str, model_path: str, labels_path: str):
	artifacts = {}
	if os.path.exists(tfidf_path):
		artifacts["tfidf"] = joblib.load(tfidf_path)
	else:
		artifacts["tfidf"] = None
	if os.path.exists(model_path):
		artifacts["model"] = joblib.load(model_path)
	else:
		artifacts["model"] = None
	if os.path.exists(labels_path):
		artifacts["label_mapping"] = joblib.load(labels_path)
	else:
		artifacts["label_mapping"] = None
	return artifacts


def predict_text(text: str, artifacts) -> Optional[dict]:
	if not text or artifacts.get("tfidf") is None or artifacts.get("model") is None or artifacts.get("label_mapping") is None:
		return None
	vec = artifacts["tfidf"].transform([text])
	model = artifacts["model"]
	pred = model.predict(vec)[0]
	proba = None
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(vec)[0]
	# map numeric label back to original label
	label_map = artifacts["label_mapping"]
	inv_map = {v: k for k, v in label_map.items()} if label_map else {}
	pred_label = inv_map.get(pred, str(pred))
	return {"label": pred_label, "proba": proba}


def main():
	st.title("Customer Support Ticket Priority — Demo")

	paths = get_paths()
	artifacts = load_artifacts(paths["tfidf"], paths["model"], paths["labels"])

	if artifacts["model"] is None or artifacts["tfidf"] is None or artifacts["label_mapping"] is None:
		st.warning("Model artifacts not found in the `models/` folder. Run training pipeline to generate them.")

	st.header("Single ticket prediction")
	text = st.text_area("Ticket description", height=160)
	if st.button("Predict"):
		res = predict_text(text, artifacts)
		if res is None:
			st.error("Missing artifacts or empty input. Ensure `models/tfidf_vectorizer.pkl`, `models/logistic_regression_model.pkl`, and `models/label_mapping.pkl` exist.")
		else:
			st.success(f"Predicted priority: {res['label']}")
			if res["proba"] is not None:
				proba_series = pd.Series(res["proba"], name="probability")
				proba_series.index = [k for k, v in sorted(artifacts["label_mapping"].items(), key=lambda kv: kv[1])]
				st.table(proba_series.to_frame())

	st.header("Batch prediction (CSV)")
	uploaded = st.file_uploader("Upload CSV (column: Ticket_Description)", type=["csv"])
	if uploaded is not None:
		df = pd.read_csv(uploaded)
		if "Ticket_Description" not in df.columns:
			st.error("CSV must contain a `Ticket_Description` column.")
		else:
			if artifacts["tfidf"] is None or artifacts["model"] is None or artifacts["label_mapping"] is None:
				st.error("Missing model artifacts.")
			else:
				texts = df["Ticket_Description"].fillna("").astype(str).tolist()
				X = artifacts["tfidf"].transform(texts)
				preds = artifacts["model"].predict(X)
				inv_map = {v: k for k, v in artifacts["label_mapping"].items()}
				pred_labels = [inv_map.get(int(p), p) for p in preds]
				df["predicted_priority"] = pred_labels
				st.dataframe(df.head(50))
				csv_out = df.to_csv(index=False).encode("utf-8")
				st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
	main()

