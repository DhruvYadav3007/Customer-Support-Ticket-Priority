import os
import re
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Support Ticket Prediction & Routing",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    .result-card {
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        color: #111827 !important;
        min-height: 110px;
    }

    .result-card .result-label {
        color: #6b7280 !important;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }

    .result-card .result-value {
        color: #111827 !important;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .result-card .priority-critical {
        color: #b91c1c !important;
    }

    .result-card .priority-high {
        color: #c2410c !important;
    }

    .result-card .priority-medium {
        color: #a16207 !important;
    }

    .result-card .priority-low {
        color: #15803d !important;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .priority-critical {
        color: #b91c1c;
        font-weight: 700;
    }

    .priority-high {
        color: #c2410c;
        font-weight: 700;
    }

    .priority-medium {
        color: #a16207;
        font-weight: 700;
    }

    .priority-low {
        color: #15803d;
        font-weight: 700;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

def get_paths():

    repo_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    models_dir = os.path.join(
        repo_root,
        "models"
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

    departments_dir = os.path.join(
        repo_root,
        "data",
        "departments"
    )

    sample_csv = os.path.join(
        repo_root,
        "sample_data",
        "sample_tickets.csv"
    )

    return {

        "model": os.path.join(
            models_dir,
            "priority_xgboost_model.pkl"
        ),

        "subject_tfidf": os.path.join(
            models_dir,
            "priority_subject_tfidf.pkl"
        ),

        "description_tfidf": os.path.join(
            models_dir,
            "priority_description_tfidf.pkl"
        ),

        "category_encoder": os.path.join(
            models_dir,
            "priority_category_encoder.pkl"
        ),

        "channel_encoder": os.path.join(
            models_dir,
            "priority_channel_encoder.pkl"
        ),

        "labels": os.path.join(
            models_dir,
            "priority_label_mapping.pkl"
        ),

        "data": data_path,

        "departments": departments_dir,

        "sample": sample_csv
    }


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts(paths):

    artifacts = {}

    names = [
        "model",
        "subject_tfidf",
        "description_tfidf",
        "category_encoder",
        "channel_encoder",
        "labels"
    ]

    for name in names:

        file_path = paths[name]

        if os.path.exists(file_path):

            artifacts[name] = joblib.load(
                file_path
            )

        else:

            artifacts[name] = None

    return artifacts


# ============================================================
# LOAD HISTORICAL DATASET
# ============================================================

@st.cache_data
def load_ticket_data(data_path):

    if not os.path.exists(data_path):
        return None

    return pd.read_csv(
        data_path
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s']",
        "",
        text
    )

    return text.strip()


# ============================================================
# NUMERIC TEXT FEATURES
# ============================================================

def text_features(data):

    subject = (
        data["Ticket_Subject"]
        .fillna("")
        .astype(str)
    )

    description = (
        data["Ticket_Description"]
        .fillna("")
        .astype(str)
    )

    features = np.column_stack([

        subject.str.len(),

        description.str.len(),

        subject.str.split().str.len(),

        description.str.split().str.len(),

        description.str.count("!"),

        description.str.count(r"\?"),

        description.str.count(
            r"\b(error|failed|failure|urgent|critical|"
            r"blocked|cannot|unable|not working|issue|problem)\b"
        )

    ])

    return features.astype(float)


# ============================================================
# LABEL MAPPING
# ============================================================

def get_number_to_label(label_mapping):

    if (
        isinstance(
            label_mapping,
            dict
        )
        and
        "number_to_label" in label_mapping
    ):

        return {
            int(k): v
            for k, v in label_mapping[
                "number_to_label"
            ].items()
        }

    return {
        int(value): key
        for key, value
        in label_mapping.items()
    }


# ============================================================
# SINGLE TICKET PREDICTION
# ============================================================

def predict_ticket(
    subject,
    description,
    category,
    channel,
    artifacts
):

    required = [
        "model",
        "subject_tfidf",
        "description_tfidf",
        "category_encoder",
        "channel_encoder",
        "labels"
    ]

    for name in required:

        if artifacts.get(name) is None:
            return None

    data = pd.DataFrame([{

        "Ticket_Subject":
            clean_text(subject),

        "Ticket_Description":
            clean_text(description),

        "Issue_Category":
            str(category),

        "Ticket_Channel":
            str(channel)

    }])

    subject_vector = (
        artifacts["subject_tfidf"]
        .transform(
            data["Ticket_Subject"]
        )
    )

    description_vector = (
        artifacts["description_tfidf"]
        .transform(
            data["Ticket_Description"]
        )
    )

    category_vector = (
        artifacts["category_encoder"]
        .transform(
            data[["Issue_Category"]]
        )
    )

    channel_vector = (
        artifacts["channel_encoder"]
        .transform(
            data[["Ticket_Channel"]]
        )
    )

    numeric_vector = text_features(
        data
    )

    X = hstack([

        subject_vector,

        description_vector,

        category_vector,

        channel_vector,

        numeric_vector

    ]).tocsr()

    model = artifacts["model"]

    prediction = model.predict(
        X
    )[0]

    prediction = int(
        prediction
    )

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )

    number_to_label = (
        get_number_to_label(
            artifacts["labels"]
        )
    )

    label = number_to_label.get(
        prediction,
        str(prediction)
    )

    return {

        "label": label,

        "proba": probabilities
    }


# ============================================================
# EXPECTED RESOLUTION RANGE
# ============================================================

def get_resolution_range(
    priority,
    category,
    df
):

    if df is None:
        return None

    required = [
        "Priority_Level",
        "Issue_Category",
        "Resolution_Time_Hours"
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return None

    subset = df[
        (df["Priority_Level"] == priority)
        &
        (df["Issue_Category"] == category)
    ][
        "Resolution_Time_Hours"
    ].dropna()

    if len(subset) < 30:

        subset = df[
            df["Priority_Level"] == priority
        ][
            "Resolution_Time_Hours"
        ].dropna()

    if len(subset) == 0:

        subset = df[
            "Resolution_Time_Hours"
        ].dropna()

    if len(subset) == 0:
        return None

    lower = subset.quantile(
        0.25
    )

    upper = subset.quantile(
        0.75
    )

    return (
        round(lower, 1),
        round(upper, 1)
    )


# ============================================================
# DEPARTMENT MAPPING
# ============================================================

def get_department(category):

    department_map = {

        "Technical":
            "Technical Support",

        "Billing":
            "Billing",

        "Fraud":
            "Fraud & Risk",

        "Account":
            "Account Support",

        "General Inquiry":
            "General Support"

    }

    return department_map.get(
        str(category),
        "General Support"
    )


# ============================================================
# DEPARTMENT FILE NAME
# ============================================================

def get_department_filename(
    department
):

    department_files = {

        "Technical Support":
            "technical.csv",

        "Billing":
            "billing.csv",

        "Fraud & Risk":
            "fraud.csv",

        "Account Support":
            "account.csv",

        "General Support":
            "general_inquiry.csv"

    }

    return department_files.get(
        department,
        "general_inquiry.csv"
    )


# ============================================================
# ROUTE TICKET
# ============================================================

def route_ticket(
    subject,
    description,
    category,
    priority,
    channel,
    resolution_range,
    departments_dir
):

    os.makedirs(
        departments_dir,
        exist_ok=True
    )

    department = get_department(
        category
    )

    filename = get_department_filename(
        department
    )

    filepath = os.path.join(
        departments_dir,
        filename
    )

    ticket_id = (
        "TKT-"
        +
        uuid.uuid4()
        .hex[:8]
        .upper()
    )

    if resolution_range:

        lower, upper = (
            resolution_range
        )

        resolution = (
            f"{lower:g}–"
            f"{upper:g} hours"
        )

    else:

        resolution = "Unavailable"

    ticket = pd.DataFrame([{

        "Ticket_ID":
            ticket_id,

        "Ticket_Subject":
            subject,

        "Ticket_Description":
            description,

        "Issue_Category":
            category,

        "Priority":
            priority,

        "Expected_Resolution":
            resolution,

        "Ticket_Channel":
            channel,

        "Routed_Department":
            department,

        "Routed_At":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    }])

    if os.path.exists(filepath):

        ticket.to_csv(
            filepath,
            mode="a",
            header=False,
            index=False
        )

    else:

        ticket.to_csv(
            filepath,
            index=False
        )

    return (
        ticket_id,
        department,
        filepath
    )


# ============================================================
# LOAD DEPARTMENT QUEUES
# ============================================================

def load_department_tickets(
    departments_dir
):

    department_files = {

        "Technical Support":
            "technical.csv",

        "Billing":
            "billing.csv",

        "Fraud & Risk":
            "fraud.csv",

        "Account Support":
            "account.csv",

        "General Support":
            "general_inquiry.csv"

    }

    department_data = {}

    for department, filename in (
        department_files.items()
    ):

        filepath = os.path.join(
            departments_dir,
            filename
        )

        if os.path.exists(filepath):

            try:

                df = pd.read_csv(
                    filepath
                )

                df.insert(
                    0,
                    "Queue No.",
                    range(
                        1,
                        len(df) + 1
                    )
                )

                department_data[
                    department
                ] = df

            except Exception:

                department_data[
                    department
                ] = pd.DataFrame()

        else:

            department_data[
                department
            ] = pd.DataFrame()

    return department_data


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    df,
    artifacts,
    ticket_df,
    departments_dir
):

    df = df.copy()

    if (
        "Ticket_Description"
        not in df.columns
    ):

        raise ValueError(
            "CSV must contain "
            "`Ticket_Description`."
        )

    if (
        "Ticket_Subject"
        not in df.columns
    ):

        df["Ticket_Subject"] = ""

    if (
        "Issue_Category"
        not in df.columns
    ):

        df["Issue_Category"] = (
            "General Inquiry"
        )

    if (
        "Ticket_Channel"
        not in df.columns
    ):

        df["Ticket_Channel"] = (
            "Web Form"
        )

    prediction_df = pd.DataFrame({

        "Ticket_Subject":
            df["Ticket_Subject"]
            .fillna("")
            .astype(str)
            .apply(clean_text),

        "Ticket_Description":
            df["Ticket_Description"]
            .fillna("")
            .astype(str)
            .apply(clean_text),

        "Issue_Category":
            df["Issue_Category"]
            .fillna(
                "General Inquiry"
            )
            .astype(str),

        "Ticket_Channel":
            df["Ticket_Channel"]
            .fillna(
                "Web Form"
            )
            .astype(str)

    })

    subject_vectors = (
        artifacts[
            "subject_tfidf"
        ].transform(
            prediction_df[
                "Ticket_Subject"
            ]
        )
    )

    description_vectors = (
        artifacts[
            "description_tfidf"
        ].transform(
            prediction_df[
                "Ticket_Description"
            ]
        )
    )

    category_vectors = (
        artifacts[
            "category_encoder"
        ].transform(
            prediction_df[
                ["Issue_Category"]
            ]
        )
    )

    channel_vectors = (
        artifacts[
            "channel_encoder"
        ].transform(
            prediction_df[
                ["Ticket_Channel"]
            ]
        )
    )

    numeric_vectors = text_features(
        prediction_df
    )

    X = hstack([

        subject_vectors,

        description_vectors,

        category_vectors,

        channel_vectors,

        numeric_vectors

    ]).tocsr()

    predictions = (
        artifacts["model"]
        .predict(X)
    )

    number_to_label = (
        get_number_to_label(
            artifacts["labels"]
        )
    )

    predicted_labels = [

        number_to_label.get(
            int(prediction),
            str(prediction)
        )

        for prediction
        in predictions

    ]

    df[
        "predicted_priority"
    ] = predicted_labels

    df[
        "routed_department"
    ] = [

        get_department(
            category
        )

        for category
        in df[
            "Issue_Category"
        ]

    ]

    resolution_ranges = []

    for priority, category in zip(

        predicted_labels,

        df[
            "Issue_Category"
        ]

    ):

        result = get_resolution_range(

            priority,

            category,

            ticket_df

        )

        if result:

            lower, upper = result

            resolution_ranges.append(

                f"{lower:g}–"
                f"{upper:g} hours"

            )

        else:

            resolution_ranges.append(
                "Unavailable"
            )

    df[
        "expected_resolution_range"
    ] = resolution_ranges

    ticket_ids = []

    for _, row in df.iterrows():

        resolution_range = None

        resolution_text = (
            row[
                "expected_resolution_range"
            ]
        )

        if (
            resolution_text
            != "Unavailable"
        ):

            try:

                parts = (
                    resolution_text
                    .replace(
                        " hours",
                        ""
                    )
                    .split("–")
                )

                resolution_range = (

                    float(parts[0]),

                    float(parts[1])

                )

            except Exception:

                resolution_range = None

        ticket_id, _, _ = route_ticket(

            row[
                "Ticket_Subject"
            ],

            row[
                "Ticket_Description"
            ],

            row[
                "Issue_Category"
            ],

            row[
                "predicted_priority"
            ],

            row[
                "Ticket_Channel"
            ],

            resolution_range,

            departments_dir

        )

        ticket_ids.append(
            ticket_id
        )

    df[
        "Ticket_ID"
    ] = ticket_ids

    return df


# ============================================================
# VALIDATE ARTIFACTS
# ============================================================

def artifacts_are_ready(
    artifacts
):

    required = [

        "model",

        "subject_tfidf",

        "description_tfidf",

        "category_encoder",

        "channel_encoder",

        "labels"

    ]

    return all(
        artifacts.get(name)
        is not None
        for name in required
    )


# ============================================================
# PRIORITY CSS CLASS
# ============================================================

def priority_class(
    priority
):

    value = str(
        priority
    ).lower()

    if value == "critical":
        return "priority-critical"

    if value == "high":
        return "priority-high"

    if value == "medium":
        return "priority-medium"

    return "priority-low"


# ============================================================
# RESULT CARD
# ============================================================

def result_card(
    label,
    value,
    css_class=""
):

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                {label}
            </div>
            <div class="result-value {css_class}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BATCH ANALYTICS
# ============================================================

def display_batch_analysis(
    batch_result,
    key_suffix="main"
):

    if batch_result is None:
        return

    st.markdown(
        "## Batch Analytics"
    )

    st.caption(
        "Overview of predictions, routing and resolution estimates."
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_tickets = len(
        batch_result
    )

    department_count = (
        batch_result[
            "routed_department"
        ].nunique()
    )

    critical_count = int(
        (
            batch_result[
                "predicted_priority"
            ]
            == "Critical"
        ).sum()
    )

    high_count = int(
        (
            batch_result[
                "predicted_priority"
            ]
            == "High"
        ).sum()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Total Tickets",
            total_tickets
        )

    with col2:

        st.metric(
            "Departments",
            department_count
        )

    with col3:

        st.metric(
            "Critical Tickets",
            critical_count
        )

    with col4:

        st.metric(
            "High Priority",
            high_count
        )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    chart_col1, chart_col2 = (
        st.columns(2)
    )

    with chart_col1:

        st.subheader(
            "Priority Distribution"
        )

        priority_counts = (
            batch_result[
                "predicted_priority"
            ]
            .value_counts()
        )

        priority_analysis = pd.DataFrame({

            "Priority":
                priority_counts.index,

            "Tickets":
                priority_counts.values

        })

        st.bar_chart(
            priority_analysis.set_index(
                "Priority"
            )
        )

    with chart_col2:

        st.subheader(
            "Department Distribution"
        )

        department_counts = (
            batch_result[
                "routed_department"
            ]
            .value_counts()
        )

        department_analysis = pd.DataFrame({

            "Department":
                department_counts.index,

            "Tickets":
                department_counts.values

        })

        st.bar_chart(
            department_analysis.set_index(
                "Department"
            )
        )

    # --------------------------------------------------------
    # PRIORITY BY DEPARTMENT
    # --------------------------------------------------------

    st.subheader(
        "Priority by Department"
    )

    priority_department = pd.crosstab(

        batch_result[
            "routed_department"
        ],

        batch_result[
            "predicted_priority"
        ]

    )

    st.dataframe(
        priority_department,
        width="stretch"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Batch Results"
    )

    display_columns = [

        "Ticket_ID",

        "Ticket_Subject",

        "Issue_Category",

        "predicted_priority",

        "expected_resolution_range",

        "routed_department",

        "Ticket_Channel"

    ]

    available_columns = [

        column

        for column
        in display_columns

        if column
        in batch_result.columns

    ]

    st.dataframe(

        batch_result[
            available_columns
        ],

        width="stretch",

        hide_index=True

    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_output = (

        batch_result
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )

    )

    st.download_button(

        "Download Predictions CSV",

        data=csv_output,

        file_name="predictions.csv",

        mime="text/csv",

        width="stretch",

        key=f"download_predictions_{key_suffix}"

    )


# ============================================================
# MAIN
# ============================================================

def main():

    paths = get_paths()

    os.makedirs(
        paths["departments"],
        exist_ok=True
    )

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if (
        "analysis"
        not in st.session_state
    ):

        st.session_state.analysis = None

    if (
        "batch_result"
        not in st.session_state
    ):

        st.session_state.batch_result = None

    if (
        "batch_source"
        not in st.session_state
    ):

        st.session_state.batch_source = None

    if (
        "batch_input_df"
        not in st.session_state
    ):

        st.session_state.batch_input_df = None

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    artifacts = load_artifacts(
        paths
    )

    ticket_df = load_ticket_data(
        paths["data"]
    )

    required_artifacts = [

        "model",

        "subject_tfidf",

        "description_tfidf",

        "category_encoder",

        "channel_encoder",

        "labels"

    ]

    missing_artifacts = [

        name

        for name
        in required_artifacts

        if artifacts.get(name)
        is None

    ]

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">'
        'Customer Support Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Predict ticket priority, estimate resolution time, '
        'and route tickets to the appropriate department.'
        '</div>',
        unsafe_allow_html=True
    )

    if missing_artifacts:

        st.warning(
            "Missing model files: "
            +
            ", ".join(
                missing_artifacts
            )
        )

    # ========================================================
    # MAIN TABS
    # ========================================================

    predict_tab, batch_tab, analytics_tab, queue_tab = (
        st.tabs(
            [
                "🎫 Predict Ticket",
                "📂 Batch Processing",
                "📊 Analytics",
                "🗂 Department Queues"
            ]
        )
    )

    # ========================================================
    # SINGLE TICKET
    # ========================================================

    with predict_tab:

        st.markdown(
            "## Analyze a New Ticket"
        )

        st.caption(
            "Enter ticket information to predict priority and routing."
        )

        left_col, right_col = (
            st.columns(
                [1.2, 1]
            )
        )

        with left_col:

            subject = st.text_input(
                "Ticket Subject",
                placeholder=(
                    "Example: Unable to access my account"
                )
            )

            description = st.text_area(
                "Ticket Description",
                height=180,
                placeholder=(
                    "Describe the customer's issue..."
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                category = st.selectbox(

                    "Issue Category",

                    [
                        "Account",
                        "Billing",
                        "Fraud",
                        "General Inquiry",
                        "Technical"
                    ]

                )

            with col2:

                channel = st.selectbox(

                    "Ticket Channel",

                    [
                        "Chat",
                        "Email",
                        "Web Form"
                    ]

                )

            analyze = st.button(

                "Analyze Ticket",

                type="primary",

                width="stretch"

            )

            if analyze:

                if not description.strip():

                    st.error(
                        "Please enter a ticket description."
                    )

                elif missing_artifacts:

                    st.error(
                        "Required model files are missing."
                    )

                else:

                    result = predict_ticket(

                        subject,

                        description,

                        category,

                        channel,

                        artifacts

                    )

                    if result:

                        priority = (
                            result["label"]
                        )

                        resolution_range = (
                            get_resolution_range(

                                priority,

                                category,

                                ticket_df

                            )
                        )

                        st.session_state.analysis = {

                            "subject":
                                subject,

                            "description":
                                description,

                            "category":
                                category,

                            "channel":
                                channel,

                            "priority":
                                priority,

                            "resolution_range":
                                resolution_range,

                            "proba":
                                result["proba"]

                        }

        with right_col:

            analysis = (
                st.session_state.analysis
            )

            if analysis is None:

                st.info(
                    "Prediction results will appear here "
                    "after you analyze a ticket."
                )

            else:

                st.markdown(
                    "## Prediction Result"
                )

                priority = analysis[
                    "priority"
                ]

                resolution = analysis[
                    "resolution_range"
                ]

                if resolution:

                    lower, upper = (
                        resolution
                    )

                    resolution_text = (
                        f"{lower:g}–"
                        f"{upper:g} hours"
                    )

                else:

                    resolution_text = (
                        "Unavailable"
                    )

                result_col1, result_col2 = (
                    st.columns(2)
                )

                with result_col1:

                    result_card(
                        "Predicted Priority",
                        priority,
                        priority_class(
                            priority
                        )
                    )

                with result_col2:

                    result_card(
                        "Expected Resolution",
                        resolution_text
                    )

                st.markdown(
                    "### Ticket Routing"
                )

                department = get_department(
                    analysis["category"]
                )

                result_card(
                    "Selected Department",
                    department
                )

                probabilities = (
                    analysis["proba"]
                )

                if probabilities is not None:

                    st.markdown(
                        "### Priority Confidence"
                    )

                    number_to_label = (
                        get_number_to_label(
                            artifacts["labels"]
                        )
                    )

                    confidence_df = pd.DataFrame({

                        "Priority": [

                            number_to_label.get(
                                int(i),
                                str(i)
                            )

                            for i
                            in range(
                                len(
                                    probabilities
                                )
                            )

                        ],

                        "Confidence (%)":

                            np.round(

                                probabilities
                                * 100,

                                2

                            )

                    })

                    st.dataframe(

                        confidence_df,

                        width="stretch",

                        hide_index=True

                    )

                if st.button(

                    "Route Ticket",

                    type="primary",

                    width="stretch",

                    key="route_single_ticket"

                ):

                    ticket_id, department, filepath = (
                        route_ticket(

                            analysis[
                                "subject"
                            ],

                            analysis[
                                "description"
                            ],

                            analysis[
                                "category"
                            ],

                            analysis[
                                "priority"
                            ],

                            analysis[
                                "channel"
                            ],

                            analysis[
                                "resolution_range"
                            ],

                            paths[
                                "departments"
                            ]

                        )
                    )

                    st.success(

                        f"Ticket {ticket_id} "
                        f"routed to {department}"

                    )

                    st.caption(

                        "Saved to: "
                        +
                        os.path.basename(
                            filepath
                        )

                    )

                    st.session_state.analysis = None

                    st.rerun()

    # ========================================================
    # BATCH PROCESSING
    # ========================================================

    with batch_tab:
        st.markdown("## Batch Ticket Processing")
        st.caption("Upload a CSV or use the included sample dataset.")

        upload_col, sample_col = st.columns(2)

        with upload_col:
            uploaded = st.file_uploader(
                "Upload CSV", type=["csv"], key="batch_upload"
            )
            if uploaded is not None:
                try:
                    st.session_state.batch_input_df = pd.read_csv(uploaded)
                    st.session_state.batch_source = uploaded.name
                except Exception as e:
                    st.error("Could not read the uploaded CSV.")
                    st.exception(e)

        with sample_col:
            st.write("Use the included demonstration dataset.")
            use_sample = st.button(
                "Use Sample CSV", width="stretch", key="use_sample_csv"
            )
            if use_sample:
                if not os.path.exists(paths["sample"]):
                    st.error(
                        "Sample CSV not found.\n\n"
                        f"Expected location:\n{paths['sample']}"
                    )
                else:
                    try:
                        st.session_state.batch_input_df = pd.read_csv(paths["sample"])
                        st.session_state.batch_source = "Included Sample CSV"
                    except Exception as e:
                        st.error("Could not read the sample CSV.")
                        st.exception(e)

        batch_df = st.session_state.get("batch_input_df")
        batch_source = st.session_state.get("batch_source")

        if batch_df is not None:
            st.markdown("### Input Preview")
            st.caption(
                f"{len(batch_df)} ticket(s) loaded from {batch_source}."
            )
            st.dataframe(batch_df.head(10), width="stretch", hide_index=True)

            if "Ticket_Description" not in batch_df.columns:
                st.error("CSV must contain `Ticket_Description`.")
            elif missing_artifacts:
                st.error("Required model files are missing.")
            else:
                process_clicked = st.button(
                    "Process Tickets",
                    type="primary",
                    width="stretch",
                    key="process_batch"
                )
                if process_clicked:
                    try:
                        with st.spinner("Processing tickets..."):
                            batch_result = process_batch(
                                batch_df.copy(),
                                artifacts,
                                ticket_df,
                                paths["departments"]
                            )

                        st.session_state.batch_result = batch_result
                        st.session_state.batch_source = batch_source
                        st.success(
                            f"{len(batch_result)} ticket(s) processed and routed "
                            f"from {batch_source}."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error("Batch processing failed.")
                        st.exception(e)

        if st.session_state.batch_result is not None:
            st.divider()
            st.markdown("## Latest Batch Results")
            if st.session_state.batch_source:
                st.caption(
                    "Source: " + str(st.session_state.batch_source)
                )
            display_batch_analysis(
                st.session_state.batch_result,
                key_suffix="batch_results"
            )

    # ========================================================
    # ANALYTICS TAB
    # ========================================================

    with analytics_tab:

        if (
            st.session_state.batch_result
            is None
        ):

            st.info(
                "Process a CSV in the Batch Processing tab "
                "to view analytics."
            )

        else:

            display_batch_analysis(

                st.session_state.batch_result

            )

    # ========================================================
    # DEPARTMENT QUEUES
    # ========================================================

    with queue_tab:

        st.markdown(
            "## Department Queues"
        )

        st.caption(
            "Tickets currently routed to each support department."
        )

        department_data = (
            load_department_tickets(
                paths[
                    "departments"
                ]
            )
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_cols = st.columns(5)

        for column, (
            department,
            department_df
        ) in zip(

            summary_cols,

            department_data.items()

        ):

            with column:

                st.metric(

                    department,

                    len(
                        department_df
                    )

                )

        st.divider()

        # ----------------------------------------------------
        # QUEUE TABS
        # ----------------------------------------------------

        tabs = st.tabs(

            list(
                department_data.keys()
            )

        )

        for tab, (
            department,
            department_df
        ) in zip(

            tabs,

            department_data.items()

        ):

            with tab:

                st.subheader(

                    f"{department} Queue"

                )

                if department_df.empty:

                    st.info(

                        "No tickets currently routed "
                        "to this department."

                    )

                else:

                    st.write(

                        f"**{len(department_df)} "
                        "current ticket(s)**"

                    )

                    display_columns = [

                        "Queue No.",

                        "Ticket_ID",

                        "Ticket_Subject",

                        "Issue_Category",

                        "Priority",

                        "Expected_Resolution",

                        "Ticket_Channel",

                        "Routed_At"

                    ]

                    available_columns = [

                        column

                        for column
                        in display_columns

                        if column
                        in department_df.columns

                    ]

                    st.dataframe(

                        department_df[
                            available_columns
                        ],

                        width="stretch",

                        hide_index=True

                    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()