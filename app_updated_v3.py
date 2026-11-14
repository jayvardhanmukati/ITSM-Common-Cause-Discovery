
from pathlib import Path
from datetime import datetime
from collections import Counter
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "itsm_final_dataset.csv"
DL_MODEL_PATH = BASE_DIR / "lstm_model.keras"
DL_METADATA_PATH = BASE_DIR / "lstm_metadata.joblib"

st.set_page_config(
    page_title="ITSM Insight | Common-Cause Discovery",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #273449;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    background: #151f32;
    border: 1px solid #28364d;
    padding: 17px;
    border-radius: 12px;
}
div[data-testid="stForm"] {
    background: #111c2e;
    padding: 20px;
    border: 1px solid #28364d;
    border-radius: 12px;
}
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #28364d;
    border-radius: 10px;
}
h1, h2, h3 {
    letter-spacing: -0.4px;
}
.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}
.status-pill {
    color: #93c5fd;
    background: #172b46;
    padding: 5px 9px;
    border-radius: 7px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------




@st.cache_resource
def load_dl_model():
    if not DL_MODEL_PATH.exists() or not DL_METADATA_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(DL_MODEL_PATH)
    metadata = joblib.load(DL_METADATA_PATH)
    names = metadata.get("class_names")
    if names is None:
        raise KeyError("lstm_metadata.joblib is missing class_names.")
    return model, [str(name) for name in names]


@st.cache_data
def load_historical_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_PATH)

    required = [
        "incident_id",
        "short_description",
        "description",
        "urgency",
        "impact",
        "category",
        "assignment_group",
        "root_cause",
        "resolution_note",
    ]

    for col in required:
        if col not in data.columns:
            data[col] = ""

    return data



dl_model, dl_class_names = load_dl_model()
historical = load_historical_data()


if dl_model is None:
    st.error("BiLSTM model files not found. Run `python model_new.py` first to create lstm_model.keras and lstm_metadata.joblib.")
    st.stop()

if historical.empty:
    st.error("Dataset not found. Please ensure `itsm_final_dataset.csv` is in the project folder.")
    st.stop()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "tickets" not in st.session_state:
    initial = historical.copy()
    initial["status"] = "Resolved"
    initial["created_at"] = "Historical"
    initial["linked_problem"] = ""
    st.session_state.tickets = initial

if "problems" not in st.session_state:
    st.session_state.problems = []

if "analysis_groups" not in st.session_state:
    st.session_state.analysis_groups = []

if "analysis_ungrouped" not in st.session_state:
    st.session_state.analysis_ungrouped = []

if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None


# --------------------------------------------------
# MODEL HELPERS
# --------------------------------------------------

# The BiLSTM makes root-cause predictions. The older TF-IDF vectorizer is
# retained only for text-similarity grouping and historical-example retrieval.



def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def incident_text(row):
    title = _clean(row.get("short_description", ""))
    description = _clean(row.get("description", ""))
    return re.sub(r"\s+", " ", f"{title} {description}").strip()


def predict_probabilities(texts):
    values = np.asarray([str(t) for t in texts], dtype=object)
    probabilities = dl_model.predict(values, batch_size=256, verbose=0)
    return np.asarray(probabilities, dtype=float)


def predict_cause(text):
    probabilities = predict_probabilities([text])[0]
    index = int(np.argmax(probabilities))
    return {
        "cause": str(dl_class_names[index]),
        "score": float(probabilities[index]),
    }


def get_lime_explanation(text, predicted_cause, num_features=7):
    """Return a local LIME explanation for one incident, if LIME is installed."""
    if LimeTextExplainer is None:
        return None
    class_index = dl_class_names.index(predicted_cause)
    explainer = LimeTextExplainer(class_names=dl_class_names)

    def predict_fn(text_list):
        return predict_probabilities(text_list)

    explanation = explainer.explain_instance(
        str(text),
        predict_fn,
        labels=[class_index],
        num_features=num_features,
        num_samples=500,
    )
    return explanation.as_list(label=class_index)


def get_similar_historical(text, limit=3):
    """Find similar historical incidents directly from the CSV dataset."""
    historical_texts = historical["short_description"].fillna("").astype(str)

    if "description" in historical.columns:
        historical_texts = (
            historical_texts + " " +
            historical["description"].fillna("").astype(str)
        )

    query_words = set(re.findall(r"\b\w+\b", str(text).lower()))

    if not query_words:
        return pd.DataFrame()

    scores = []

    for historical_text in historical_texts:
        words = set(re.findall(r"\b\w+\b", historical_text.lower()))

        union = query_words | words
        similarity = len(query_words & words) / len(union) if union else 0.0
        scores.append(similarity)

    results = historical.copy()
    results["similarity"] = scores

    results = results.sort_values("similarity", ascending=False)
    results = results[results["similarity"] > 0].head(limit)

    return results.reset_index(drop=True)


def suggested_team(cause, group_rows):
    """Suggest an assignment team using historical dataset records."""
    if "root_cause" not in historical.columns:
        return "IT Service Desk"

    matches = historical[
        historical["root_cause"].astype(str).str.strip().str.lower()
        == str(cause).strip().lower()
    ]

    if not matches.empty and "assignment_group" in matches.columns:
        teams = matches["assignment_group"].dropna().astype(str).str.strip()

        teams = teams[
            (teams != "") &
            (teams.str.lower() != "nan")
        ]

        if not teams.empty:
            return teams.mode().iloc[0]

    return "IT Service Desk"


def make_groups(rows, min_confidence=0.50, min_group_size=2):
    """
    Group incidents by the BiLSTM's predicted root cause.
    All incidents are scored in ONE batch; incidents predicted to share the
    same root cause (with confidence >= min_confidence) form one group.
    Returns (groups, ungrouped).
    """
    if len(rows) == 0:
        return [], []

    texts = [incident_text(row) for row in rows]
    probabilities = predict_probabilities(texts)
    best_idx = probabilities.argmax(axis=1)
    best_conf = probabilities.max(axis=1)

    buckets = {}
    ungrouped = []
    for row, idx, conf in zip(rows, best_idx, best_conf):
        member = dict(row)
        member["predicted_cause"] = dl_class_names[int(idx)]
        member["model_confidence"] = float(conf)
        if conf >= min_confidence:
            buckets.setdefault(member["predicted_cause"], []).append(member)
        else:
            ungrouped.append(member)

    groups = []
    for cause, members in buckets.items():
        if len(members) < min_group_size:
            ungrouped.extend(members)
            continue
        members.sort(key=lambda m: m["model_confidence"], reverse=True)
        avg_conf = sum(m["model_confidence"] for m in members) / len(members)

        # Verified historical examples with the same root cause, as evidence.
        hist = historical[
            historical["root_cause"].astype(str).str.strip().str.lower()
            == cause.strip().lower()
        ]
        if "resolution_note" in hist.columns:
            with_notes = hist[hist["resolution_note"].fillna("").astype(str).str.strip() != ""]
            if not with_notes.empty:
                hist = with_notes
        related_history = hist.head(3).reset_index(drop=True)

        groups.append({
            "members": members,
            "cause": cause,
            "team": suggested_team(cause, members),
            "average_score": avg_conf,
            "review_status": (
                "Review suggested cause"
                if avg_conf < 0.70
                else "Proposed cause — needs verification"
            ),
            "related_history": related_history,
        })

    groups.sort(key=lambda g: (len(g["members"]), g["average_score"]), reverse=True)
    return groups, ungrouped


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🔎 ITSM Insight")
    st.caption("Common-Cause Discovery Platform")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Incident Queue",
            "Common-Cause Discovery",
            "Problem Management",
            "Model Evaluation",
        ],
    )

    st.divider()
    st.caption("MODEL STATUS")
    st.success("Trained model loaded")
    st.caption("BiLSTM deep-learning classifier")
    st.caption("Grouping by predicted root cause")
    st.divider()
    st.caption("Academic prototype · Local demo")


# --------------------------------------------------
# COMMON DATA
# --------------------------------------------------

tickets = st.session_state.tickets
new_count = int((tickets["status"] == "New").sum())
active_count = int(
    tickets["status"].isin(["New", "In Progress", "Under Investigation"]).sum()
)
problem_count = len(st.session_state.problems)

st.markdown(
    '<p class="small-muted">IT SERVICE MANAGEMENT / COMMON-CAUSE DISCOVERY</p>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.title("IT Operations Overview")
    st.write(
        "Monitor incidents, discover related problems and coordinate "
        "investigation across support teams."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("All tickets", f"{len(tickets):,}")
    col2.metric("New incidents", f"{new_count:,}")
    col3.metric("Active incidents", f"{active_count:,}")
    col4.metric("Problem records", f"{problem_count:,}")

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Incidents by category")
        category_counts = (
            tickets["category"].fillna("Unknown").value_counts()
            .rename_axis("Category")
            .reset_index(name="Incidents")
        )

        fig = px.bar(
            category_counts,
            x="Category",
            y="Incidents",
            color="Incidents",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Incident urgency")
        urgency_counts = (
            tickets["urgency"].fillna("Unknown").value_counts()
            .rename_axis("Urgency")
            .reset_index(name="Incidents")
        )

        fig = px.pie(
            urgency_counts,
            names="Urgency",
            values="Incidents",
            hole=0.55,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recently added incidents")
    recent = tickets[tickets["created_at"] != "Historical"].copy()

    if recent.empty:
        st.info(
            "No new incidents yet. Open Incident Queue and add a test incident."
        )
    else:
        st.dataframe(
            recent[
                [
                    "incident_id",
                    "short_description",
                    "urgency",
                    "status",
                    "assignment_group",
                ]
            ].tail(10),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "The imported dataset contains synthetic historical incidents. "
        "Dashboard counts are dataset counts, not live production IT operations."
    )


# --------------------------------------------------
# INCIDENT QUEUE
# --------------------------------------------------

elif page == "Incident Queue":
    st.title("Incident Queue")
    st.write("Register, inspect and track IT support incidents.")

    with st.expander("＋ Register a new incident", expanded=True):
        with st.form("new_incident_form", clear_on_submit=True):
            title = st.text_input(
                "Short description",
                placeholder="e.g. Multiple users cannot connect to VPN",
            )

            description = st.text_area(
                "Detailed description",
                placeholder="What is happening, who is affected, and when did it start?",
                height=110,
            )

            c1, c2, c3 = st.columns(3)

            urgency = c1.selectbox(
                "Urgency",
                ["1 - High", "2 - Medium", "3 - Low"],
                index=1,
            )
            impact = c2.selectbox(
                "Impact",
                ["1 - High", "2 - Medium", "3 - Low"],
                index=1,
            )
            category = c3.selectbox(
                "Category",
                [
                    "Network",
                    "Database",
                    "Access",
                    "Storage",
                    "Performance",
                    "Application",
                    "Security",
                    "Email",
                    "Software",
                    "Other",
                ],
            )

            submitted = st.form_submit_button(
                "Create incident",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                if not title.strip() or not description.strip():
                    st.error("Please enter both a short and detailed description.")
                else:
                    existing_ids = tickets["incident_id"].astype(str)
                    numbers = [
                        int(match.group(1))
                        for item in existing_ids
                        if (match := re.search(r"INC(\d+)", item))
                    ]
                    next_number = max(numbers, default=0) + 1
                    new_id = f"INC{next_number:05d}"

                    # A new ticket is registered first; the model analyzes it
                    # during the common-cause discovery stage.
                    new_row = {
                        "incident_id": new_id,
                        "short_description": title.strip(),
                        "description": description.strip(),
                        "urgency": urgency,
                        "impact": impact,
                        "category": category,
                        "assignment_group": "Unassigned",
                        "root_cause": "Not verified",
                        "resolution_note": "",
                        "status": "New",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "linked_problem": "",
                    }

                    st.session_state.tickets = pd.concat(
                        [
                            st.session_state.tickets,
                            pd.DataFrame([new_row]),
                        ],
                        ignore_index=True,
                    )
                    st.session_state.analysis_groups = []
                    st.session_state.analysis_ungrouped = []
                    st.success(f"Incident {new_id} created successfully.")
                    st.rerun()

    st.divider()

    current = st.session_state.tickets.copy()
    filter1, filter2, filter3 = st.columns(3)

    status_options = ["All"] + sorted(
        current["status"].fillna("Unknown").unique().tolist()
    )
    selected_status = filter1.selectbox("Filter by status", status_options)

    urgency_options = ["All"] + sorted(
        current["urgency"].fillna("Unknown").unique().tolist()
    )
    selected_urgency = filter2.selectbox("Filter by urgency", urgency_options)

    search_text = filter3.text_input("Search tickets", placeholder="Incident ID or title")

    if selected_status != "All":
        current = current[current["status"] == selected_status]

    if selected_urgency != "All":
        current = current[current["urgency"] == selected_urgency]

    if search_text.strip():
        query = search_text.strip().lower()
        mask = (
            current["incident_id"].astype(str).str.lower().str.contains(query, na=False)
            | current["short_description"].astype(str).str.lower().str.contains(query, na=False)
        )
        current = current[mask]

    st.caption(f"Showing {len(current)} incidents")

    st.dataframe(
        current[
            [
                "incident_id",
                "short_description",
                "urgency",
                "impact",
                "category",
                "status",
                "assignment_group",
                "linked_problem",
            ]
        ].sort_values("incident_id", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------
# COMMON-CAUSE DISCOVERY
# --------------------------------------------------

elif page == "Common-Cause Discovery":
    st.title("Common-Cause Discovery")
    st.write(
        "The BiLSTM reads every selected incident and groups the ones that "
        "point to the same underlying root cause."
    )

    st.markdown("### 1. Select the incidents to analyze")

    active = st.session_state.tickets.copy()

    if active.empty:
        st.info("There are no incidents available to analyze. Add incidents in Incident Queue.")
    else:
        is_new = active["created_at"].astype(str) != "Historical"
        new_df = active[is_new]
        hist_df = active[~is_new]

        s1, s2, s3 = st.columns([1.4, 1, 1])
        mode = s1.radio(
            "Incident source",
            [
                "New incidents only",
                "Sample of historical incidents",
                "New + sample of historical",
            ],
            index=0 if len(new_df) else 1,
        )
        sample_size = s2.slider(
            "Historical sample size", 20, 500, 100, 10,
            help="Used when historical incidents are included.",
        )
        min_conf = s3.slider(
            "Minimum model confidence", 0.30, 0.95, 0.50, 0.05,
            help="Incidents below this confidence are not grouped.",
        )
        min_size = st.slider(
            "Minimum incidents per group", 2, 10, 2, 1,
            help="A root cause needs at least this many incidents to form a card.",
        )

        if mode == "New incidents only":
            chosen_df = new_df
        elif mode == "Sample of historical incidents":
            chosen_df = hist_df.sample(min(sample_size, len(hist_df)), random_state=7)
        else:
            chosen_df = pd.concat(
                [new_df, hist_df.sample(min(sample_size, len(hist_df)), random_state=7)]
            )

        st.caption(f"{len(chosen_df)} incidents will be analyzed.")

        if st.button("Analyze incidents together", type="primary", use_container_width=True):
            if chosen_df.empty:
                st.warning("No incidents to analyze. Add an incident or choose a historical sample.")
            else:
                with st.spinner("Running BiLSTM on selected incidents..."):
                    g, u = make_groups(
                        chosen_df.to_dict("records"), min_conf, min_size
                    )
                st.session_state.analysis_groups = g
                st.session_state.analysis_ungrouped = u
                st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()

        groups = st.session_state.analysis_groups
        ungrouped = st.session_state.get("analysis_ungrouped", [])

        if groups or ungrouped:
            st.divider()
            st.markdown("### 2. Discovery results")

            grouped_total = sum(len(g["members"]) for g in groups)
            c1, c2, c3 = st.columns(3)
            c1.metric("Common-cause groups", len(groups))
            c2.metric("Incidents grouped", grouped_total)
            c3.metric("Ungrouped / low confidence", len(ungrouped))
            st.caption(f"Last analyzed: {st.session_state.analysis_time or 'Not available'}")

            if not groups:
                st.info(
                    "No root cause had enough confident incidents. Lower the "
                    "confidence threshold or minimum group size and analyze again."
                )

            for index, group in enumerate(groups, start=1):
                members = group["members"]
                cause = group["cause"]

                with st.container(border=True):
                    st.caption(f"GROUP {index}")
                    st.markdown("**Proposed root cause**")
                    st.subheader(cause)
                    st.caption(group["review_status"])

                    a, b, c = st.columns(3)
                    a.metric("Incidents in group", len(members))
                    b.metric("Average model confidence", f"{group['average_score']:.0%}")
                    c.metric("Suggested team", group["team"])

                    st.markdown("**Incidents with this root cause**")
                    member_df = pd.DataFrame(members).fillna("")
                    member_df["Confidence"] = member_df["model_confidence"].map(lambda x: f"{x:.0%}")
                    cols = [c_ for c_ in
                            ["incident_id", "short_description", "urgency", "category", "Confidence"]
                            if c_ in member_df.columns]
                    st.dataframe(member_df[cols], use_container_width=True, hide_index=True)

                    history = group["related_history"]
                    if not history.empty:
                        with st.expander("Verified historical incidents with this root cause"):
                            hcols = [c_ for c_ in
                                     ["incident_id", "short_description", "root_cause", "resolution_note"]
                                     if c_ in history.columns]
                            st.dataframe(history[hcols].fillna(""),
                                         use_container_width=True, hide_index=True)

                    with st.expander("Explain the model prediction (LIME)"):
                        if LimeTextExplainer is None:
                            st.info("Install LIME with: python -m pip install lime")
                        else:
                            st.caption(
                                "Local explanation for the top-confidence incident in this group. "
                                "Highlighted words influence the prediction; they do not prove causation."
                            )
                            if st.button("Generate explanation", key=f"lime_{index}"):
                                with st.spinner("Generating explanation..."):
                                    lime_items = get_lime_explanation(incident_text(members[0]), cause)
                                if lime_items:
                                    st.dataframe(
                                        pd.DataFrame(lime_items, columns=["Text feature", "Influence on predicted cause"]),
                                        use_container_width=True, hide_index=True,
                                    )

                    if group["average_score"] < 0.70:
                        st.warning("Average model confidence is modest. Review before linking these tickets.")

            if ungrouped:
                with st.expander(f"Ungrouped / low-confidence incidents ({len(ungrouped)})"):
                    u_df = pd.DataFrame(ungrouped).fillna("")
                    u_df["Best guess"] = u_df["predicted_cause"]
                    u_df["Confidence"] = u_df["model_confidence"].map(lambda x: f"{x:.0%}")
                    ucols = [c_ for c_ in
                             ["incident_id", "short_description", "Best guess", "Confidence"]
                             if c_ in u_df.columns]
                    st.dataframe(u_df[ucols], use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### 3. Review and send to IT")
            st.write(
                "Review the proposed cause and assignment, then send the group to the responsible team. "
                "Sending is simulated locally; no real email or ServiceNow notification is sent."
            )

            if st.button("Send reviewed groups to IT", type="primary", use_container_width=True):
                created = 0

                for group in groups:
                    members = group["members"]
                    if len(members) < 2:
                        continue
                    if all(str(row.get("created_at", "")) == "Historical" for row in members):
                        continue
                    cause = group["cause"]
                    team = group["team"]
                    ids = [str(row["incident_id"]) for row in members]

                    existing_sets = [set(p["incident_ids"]) for p in st.session_state.problems]
                    if set(ids) in existing_sets:
                        continue

                    problem_id = f"PRB{len(st.session_state.problems) + 1:05d}"
                    st.session_state.problems.append({
                        "problem_id": problem_id,
                        "title": f"Investigate: {cause}",
                        "proposed_cause": cause,
                        "assignment_group": team,
                        "incident_ids": ids,
                        "status": "Assigned",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "verified_cause": "",
                        "resolution": "",
                    })

                    mask = st.session_state.tickets["incident_id"].astype(str).isin(ids)
                    st.session_state.tickets.loc[mask, "linked_problem"] = problem_id
                    new_mask = mask & (st.session_state.tickets["created_at"].astype(str) != "Historical")
                    st.session_state.tickets.loc[new_mask, "status"] = "In Progress"
                    st.session_state.tickets.loc[new_mask, "assignment_group"] = team
                    created += 1

                if created:
                    st.success(
                        f"Sent {created} problem group(s) to the assigned IT team in the demo workflow. "
                        "Open Problem Management to track them."
                    )
                    st.rerun()
                else:
                    st.info("No new groups to send (groups made only of historical incidents are reference-only).")

    st.divider()

# --------------------------------------------------
# PROBLEM MANAGEMENT
# --------------------------------------------------

elif page == "Problem Management":
    st.title("Problem Management")
    st.write(
        "Track assigned problem groups, update investigation notes and resolve linked incidents."
    )

    problems = st.session_state.problems

    if not problems:
        st.info(
            "No problem records yet. Analyze incidents and create problem records "
            "from the Common-Cause Discovery page."
        )
    else:
        for index, problem in enumerate(problems):
            with st.container(border=True):
                st.markdown(
                    f"### {problem['problem_id']} · {problem['title']}"
                )

                a, b, c = st.columns(3)
                a.metric("Linked incidents", len(problem["incident_ids"]))
                b.metric("Assignment group", problem["assignment_group"])
                c.metric("Status", problem["status"])

                st.write(
                    "**Model-proposed cause:**",
                    problem["proposed_cause"],
                )
                st.write(
                    "**Linked ticket IDs:**",
                    ", ".join(problem["incident_ids"]),
                )

                with st.form(f"resolve_{problem['problem_id']}"):
                    st.markdown("**IT investigation and resolution**")

                    verified = st.selectbox(
                        "Verified root cause",
                        [
                            "Not yet verified",
                            problem["proposed_cause"],
                            "Different cause identified",
                        ],
                        key=f"verified_{problem['problem_id']}",
                    )

                    actual_cause = st.text_input(
                        "Corrected root cause (if different)",
                        value=problem["verified_cause"],
                        key=f"actual_{problem['problem_id']}",
                    )

                    resolution = st.text_area(
                        "Investigation notes / fix",
                        value=problem["resolution"],
                        key=f"resolution_{problem['problem_id']}",
                    )

                    status = st.selectbox(
                        "Problem status",
                        [
                            "Awaiting human review",
                            "Assigned",
                            "Investigating",
                            "Resolved",
                            "Closed",
                        ],
                        index=(
                            [
                                "Awaiting human review",
                                "Assigned",
                                "Investigating",
                                "Resolved",
                                "Closed",
                            ].index(problem["status"])
                            if problem["status"] in [
                                "Awaiting human review",
                                "Assigned",
                                "Investigating",
                                "Resolved",
                                "Closed",
                            ]
                            else 0
                        ),
                        key=f"status_{problem['problem_id']}",
                    )

                    save = st.form_submit_button("Save update / mark resolved")

                    if save:
                        problem["status"] = status
                        problem["resolution"] = resolution

                        if verified == "Different cause identified" and actual_cause.strip():
                            problem["verified_cause"] = actual_cause.strip()
                        elif verified == problem["proposed_cause"]:
                            problem["verified_cause"] = problem["proposed_cause"]
                        else:
                            problem["verified_cause"] = ""

                        if status in ["Resolved", "Closed"]:
                            ids = problem["incident_ids"]
                            mask = st.session_state.tickets["incident_id"].astype(str).isin(ids)
                            st.session_state.tickets.loc[mask, "status"] = status

                        st.success(f"{problem['problem_id']} updated.")
                        st.rerun()


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write(
        "Held-out evaluation of the saved BiLSTM classifier. "
        "Results are based on the evaluation report generated by model_new.py."
    )

    evaluation_path = BASE_DIR / "lstm_evaluation.json"
    report_path = BASE_DIR / "classification_report.csv"
    matrix_path = BASE_DIR / "confusion_matrix.csv"
    evaluation = {}
    if evaluation_path.exists():
        try:
            import json
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except Exception:
            evaluation = {}

    accuracy = float(evaluation.get("accuracy", 0))
    report = evaluation.get("classification_report", evaluation.get("report", {}))
    report.setdefault("weighted avg", {
        "precision": float(evaluation.get("precision_weighted", 0)),
        "recall": float(evaluation.get("recall_weighted", 0)),
        "f1-score": float(evaluation.get("f1_weighted", 0)),
    })
    report.setdefault("macro avg", {
        "precision": float(evaluation.get("precision_macro", 0)),
        "recall": float(evaluation.get("recall_macro", 0)),
        "f1-score": float(evaluation.get("f1_macro", 0)),
    })
    labels = list(dl_class_names)
    matrix = None
    if matrix_path.exists():
        try:
            matrix = pd.read_csv(matrix_path, index_col=0).to_numpy()
        except Exception:
            matrix = None

    if report_path.exists():
        try:
            report_df_from_file = pd.read_csv(report_path, index_col=0)
            if {"precision", "recall", "f1-score"}.issubset(report_df_from_file.columns):
                report.update({
                    str(label): {
                        "precision": float(row.get("precision", 0)),
                        "recall": float(row.get("recall", 0)),
                        "f1-score": float(row.get("f1-score", 0)),
                        "support": float(row.get("support", 0)),
                    }
                    for label, row in report_df_from_file.iterrows()
                    if str(label) not in {"accuracy", "macro avg", "weighted avg"}
                })
        except Exception:
            pass

    weighted = report.get("weighted avg", {})
    macro = report.get("macro avg", {})

    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{accuracy:.1%}")
    b.metric("Weighted precision", f"{weighted.get('precision', 0):.1%}")
    c.metric("Weighted recall", f"{weighted.get('recall', 0):.1%}")
    d.metric("Weighted F1-score", f"{weighted.get('f1-score', 0):.1%}")

    st.caption(
        "These metrics measure individual incident classification, not "
        "whether the system discovers the true common root cause of a group."
    )

    st.subheader("Classification report")

    rows = []
    for label in labels:
        item = report.get(label, {})
        rows.append({
            "Root cause": label,
            "Precision": item.get("precision", 0),
            "Recall": item.get("recall", 0),
            "F1-score": item.get("f1-score", 0),
            "Support": item.get("support", 0),
        })

    report_df = pd.DataFrame(rows)

    if not report_df.empty:
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-score": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Confusion matrix")

    if matrix is not None and len(labels) > 0:
        fig = px.imshow(
            matrix,
            x=labels,
            y=labels,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted root cause",
                "y": "Actual root cause",
                "color": "Count",
            },
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=20, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)

