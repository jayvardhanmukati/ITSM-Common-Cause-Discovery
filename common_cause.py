"""
Common-cause candidate discovery for the ITSM BiLSTM proof of concept.

Run from the project folder:
    python common_cause.py
    python common_cause.py --input my_incidents.csv

CSV input columns:
    short_description, description
Optional:
    incident_id, urgency, impact, category

This groups incidents with matching predicted root-cause labels. These are
candidate groups for human review, NOT proof that incidents share a true cause.
"""
from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.keras"
METADATA_PATH = BASE_DIR / "lstm_metadata.joblib"

# Recommendations are rule-based suggestions for the college POC.
RECOMMENDATIONS = {
    "Access Permission Misconfiguration": (
        "Verify the affected user's roles, permissions, and access policies; correct the missing or incorrect permission.",
        "Access Management / IT Support",
    ),
    "Application Service Failure": (
        "Check application health, service logs, dependencies, and recent deployments; restart or roll back only if appropriate.",
        "Application Support",
    ),
    "DNS Resolution Failure": (
        "Check DNS records, resolver reachability, cache, and hostname configuration; correct invalid records or resolver settings.",
        "Network Support",
    ),
    "Database Connectivity Failure": (
        "Verify database availability, connection strings, network reachability, credentials, and connection-pool limits.",
        "Database / DBA Team",
    ),
    "Disk Space Exhaustion": (
        "Check disk usage and logs; safely clear approved temporary files or expand storage, then monitor capacity.",
        "Infrastructure / Server Support",
    ),
    "Email Server Configuration Error": (
        "Check mail-server settings, SMTP connectivity, DNS/MX records, authentication, and mail-flow logs.",
        "Messaging / Email Support",
    ),
    "Expired SSL Certificate": (
        "Verify certificate expiry and certificate chain; renew and deploy the valid certificate, then test the endpoint.",
        "Security / Certificate Management",
    ),
    "Memory Resource Exhaustion": (
        "Inspect memory usage and process logs; identify abnormal consumers and scale or tune resources as approved.",
        "Infrastructure / Server Support",
    ),
    "Software Update Compatibility Issue": (
        "Review the recent update and compatibility logs; test a supported patch or rollback in line with change procedures.",
        "Endpoint / Application Support",
    ),
    "VPN Configuration Error": (
        "Verify VPN profile, gateway, authentication, routes, and client configuration; apply the approved corrected settings.",
        "Network Support",
    ),
}

DEMO_INCIDENTS = [
    ("INC-DEMO-001", "Cannot connect to database", "Application reports database connection timeout and cannot load customer records."),
    ("INC-DEMO-002", "Database connection timeout", "Several requests fail because the application cannot establish a database connection."),
    ("INC-DEMO-003", "Customer portal data unavailable", "Portal shows an error when loading data from the backend database."),
    ("INC-DEMO-004", "DB connection pool exhausted", "Application logs show failed database connections and connection-pool timeouts."),
    ("INC-DEMO-005", "Order service cannot reach database", "Order processing fails with a database connectivity error."),
    ("INC-DEMO-006", "Reports fail to load", "Reporting service times out while connecting to the database server."),
    ("INC-DEMO-007", "Backend database unavailable", "Users receive errors because the backend service cannot connect to the database."),
    ("INC-DEMO-008", "VPN login fails", "Remote staff cannot establish a VPN connection after a profile configuration change."),
    ("INC-DEMO-009", "SSL certificate warning", "Browser reports that the service certificate has expired."),
    ("INC-DEMO-010", "Disk nearly full", "Server reports low disk space and application logs cannot be written."),
]


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun model_new.py first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}\nRun model_new.py first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    class_names = metadata.get("class_names")
    if class_names is None:
        raise KeyError("lstm_metadata.joblib does not contain 'class_names'.")
    return model, [str(x) for x in class_names]


def read_incidents(input_path):
    df = pd.read_csv(input_path).fillna("")
    for col in ("short_description", "description"):
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain a '{col}' column.")
    if "incident_id" not in df.columns:
        df.insert(0, "incident_id", [f"INC-{i+1:03d}" for i in range(len(df))])
    for col in ("urgency", "impact", "category"):
        if col not in df.columns:
            df[col] = ""
    return df


def analyze_incidents(records, model, class_names, confidence_threshold=0.60, min_group_size=2):
    df = pd.DataFrame(records).fillna("")
    for col in ("short_description", "description"):
        if col not in df.columns:
            df[col] = ""
    if "incident_id" not in df.columns:
        df.insert(0, "incident_id", [f"INC-{i+1:03d}" for i in range(len(df))])

    texts = (
        df["short_description"].astype(str).str.strip()
        + " "
        + df["description"].astype(str).str.strip()
    ).str.strip().tolist()

    if not texts:
        raise ValueError("No incidents supplied.")
    probabilities = model.predict(np.asarray(texts, dtype=object), verbose=0)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Model output has {probabilities.shape[-1]} classes but metadata has "
            f"{len(class_names)} class names. Check lstm_metadata.joblib."
        )

    predicted_indices = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    df["predicted_root_cause"] = [class_names[i] for i in predicted_indices]
    df["model_confidence"] = confidences.round(4)
    df["confidence_meets_threshold"] = confidences >= confidence_threshold

    group_rows = []
    for cause, group in df[df["confidence_meets_threshold"]].groupby("predicted_root_cause"):
        if len(group) < min_group_size:
            continue
        resolution, team = RECOMMENDATIONS.get(
            str(cause),
            ("Review logs and configuration for this predicted issue before selecting a fix.", "IT Support"),
        )
        group_id = f"CC-{len(group_rows)+1:03d}"
        for _, row in group.iterrows():
            group_rows.append({
                "candidate_group_id": group_id,
                "incident_id": row["incident_id"],
                "short_description": row.get("short_description", ""),
                "description": row.get("description", ""),
                "predicted_root_cause": cause,
                "model_confidence": float(row["model_confidence"]),
                "group_size": int(len(group)),
                "suggested_resolution": resolution,
                "suggested_team": team,
                "review_status": "Needs human review",
            })

    groups_df = pd.DataFrame(group_rows)
    if not groups_df.empty:
        summary_df = groups_df.groupby(
            ["candidate_group_id", "predicted_root_cause", "group_size", "suggested_resolution", "suggested_team", "review_status"],
            as_index=False
        ).agg(
            incident_ids=("incident_id", lambda x: ", ".join(map(str, x))),
            average_confidence=("model_confidence", "mean"),
        )
        summary_df["average_confidence"] = summary_df["average_confidence"].round(4)
    else:
        summary_df = pd.DataFrame(columns=[
            "candidate_group_id", "predicted_root_cause", "group_size", "suggested_resolution",
            "suggested_team", "review_status", "incident_ids", "average_confidence"
        ])

    return df, groups_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Find candidate common-cause groups with the saved BiLSTM model.")
    parser.add_argument("--input", help="CSV containing short_description and description columns.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Minimum model confidence (default: 0.60).")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum incidents per candidate group (default: 2).")
    parser.add_argument("--demo", action="store_true", help="Use 10 built-in demo incidents.")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")
    if args.min_group_size < 2:
        parser.error("--min-group-size must be at least 2.")
    if args.input and args.demo:
        parser.error("Use either --input or --demo, not both.")

    try:
        model, class_names = load_assets()
        if args.input:
            incidents = read_incidents(args.input)
            source = f"CSV input: {args.input}"
        else:
            incidents = pd.DataFrame(DEMO_INCIDENTS, columns=["incident_id", "short_description", "description"])
            incidents["urgency"] = ""
            incidents["impact"] = ""
            incidents["category"] = ""
            source = "built-in demo incidents" if args.demo or not args.input else "demo"

        details, group_members, group_summary = analyze_incidents(
            incidents, model, class_names, args.threshold, args.min_group_size
        )

        details_path = BASE_DIR / "common_cause_results.csv"
        members_path = BASE_DIR / "common_cause_group_members.csv"
        summary_path = BASE_DIR / "common_cause_summary.json"
        details.to_csv(details_path, index=False)
        group_members.to_csv(members_path, index=False)
        payload = {
            "source": source,
            "confidence_threshold": args.threshold,
            "minimum_group_size": args.min_group_size,
            "incidents_analyzed": int(len(details)),
            "candidate_groups_found": int(group_summary.shape[0]),
            "groups": group_summary.to_dict(orient="records"),
            "note": "Candidate groups are based on matching model predictions and confidence threshold. They are not proof of a shared real-world cause and require IT review."
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print("\n=== ITSM COMMON-CAUSE CANDIDATE DISCOVERY ===")
        print(f"Input source: {source}")
        print(f"Incidents analyzed: {len(details)}")
        print(f"Candidate groups found: {len(group_summary)}")
        print("\nIncident predictions:")
        print(details[["incident_id", "short_description", "predicted_root_cause", "model_confidence"]].to_string(index=False))
        print("\nCandidate group summaries:")
        if group_summary.empty:
            print("No groups met the confidence threshold and minimum group size.")
            print("Try --threshold 0.50 for exploration; review the results carefully.")
        else:
            print(group_summary.to_string(index=False))
        print("\nSaved:")
        print(f"  {details_path.name}")
        print(f"  {members_path.name}")
        print(f"  {summary_path.name}")
        print("\nReminder: review candidate groups before sending them to IT.")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
