"""
Deep Learning-Based Common-Cause Discovery in ITSM
--------------------------------------------------
Model training/evaluation script for the ITSM project.

Input:
    itsm_final_dataset.csv (in the same folder as this script)

Expected minimum columns:
    short_description, description, root_cause

Optional useful columns:
    incident_group_id, suggested_split, source, is_synthetic, incident_id

Outputs:
    lstm_model.keras
    lstm_metadata.joblib
    lstm_evaluation.json
    preprocessing_summary.json
    classification_report.csv
    confusion_matrix.csv
    confusion_matrix.png
    training_history.png
    lstm_lime_explanation.html
    test_predictions.csv
    candidate_common_cause_groups.csv

Important:
- This is a text classifier, not proof of causation.
- Candidate groups are formed from the model's predicted root-cause labels.
  They must be reviewed by a human and evaluated separately before claiming
  that the system has discovered a real shared cause.
- The script does not edit app.py or the older model_bundle.joblib.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import TextVectorization

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "itsm_final_dataset.csv"
MODEL_PATH = BASE_DIR / "lstm_model.keras"
METADATA_PATH = BASE_DIR / "lstm_metadata.joblib"
EVALUATION_PATH = BASE_DIR / "lstm_evaluation.json"
PREPROCESSING_PATH = BASE_DIR / "preprocessing_summary.json"
REPORT_PATH = BASE_DIR / "classification_report.csv"
CM_CSV_PATH = BASE_DIR / "confusion_matrix.csv"
CM_PNG_PATH = BASE_DIR / "confusion_matrix.png"
HISTORY_PNG_PATH = BASE_DIR / "training_history.png"
LIME_PATH = BASE_DIR / "lstm_lime_explanation.html"
PREDICTIONS_PATH = BASE_DIR / "test_predictions.csv"
GROUPS_PATH = BASE_DIR / "candidate_common_cause_groups.csv"

REQUIRED_COLUMNS = ["short_description", "description", "root_cause"]
TEXT_SEQUENCE_LENGTH = 120
MAX_TOKENS = 10000
EMBEDDING_DIM = 64
BATCH_SIZE = 32
MAX_EPOCHS = 20


def json_safe(value):
    """Convert NumPy/Python values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def load_and_preprocess():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that itsm_final_dataset.csv is in the same folder as model.py."
        )

    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)
    original_columns = list(df.columns)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}\n"
            f"Columns found: {original_columns}"
        )

    summary = {
        "dataset_file": DATA_PATH.name,
        "rows_at_start": int(original_rows),
        "columns_found": original_columns,
        "missing_values_before_cleaning": {
            c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS
        },
    }

    # Convert required text fields to strings while preserving missing values.
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].where(df[col].notna(), "").astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "None": ""})

    # Classification requires a known root-cause label and some text.
    df["model_text"] = (
        df["short_description"].fillna("") + " " + df["description"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    empty_text_rows = int(df["model_text"].eq("").sum())
    missing_label_rows = int(df["root_cause"].eq("").sum())
    before_filter = len(df)
    df = df[(df["model_text"] != "") & (df["root_cause"] != "")].copy()
    rows_after_required_filter = len(df)

    # Remove exact duplicate text examples. If the same text appears with
    # different labels, exclude the conflicting text rather than choose a label.
    duplicate_label_counts = df.groupby("model_text")["root_cause"].nunique()
    conflicting_texts = set(duplicate_label_counts[duplicate_label_counts > 1].index)
    conflicting_rows = int(df["model_text"].isin(conflicting_texts).sum())
    if conflicting_texts:
        df = df[~df["model_text"].isin(conflicting_texts)].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["model_text"], keep="first").copy()
    exact_duplicate_rows_removed = before_dedup - len(df)

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} labelled, non-empty, unique examples remain. "
            "Please inspect the dataset before training."
        )

    class_counts = df["root_cause"].value_counts().sort_index().to_dict()
    summary.update({
        "empty_text_rows_removed": empty_text_rows,
        "rows_without_root_cause_label_excluded_from_supervised_training": missing_label_rows,
        "rows_after_required_field_filter": rows_after_required_filter,
        "conflicting_text_rows_excluded": conflicting_rows,
        "exact_duplicate_text_rows_removed": int(exact_duplicate_rows_removed),
        "rows_used_for_supervised_training": int(len(df)),
        "root_cause_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "preprocessing_steps": [
            "Loaded CSV",
            "Checked required columns",
            "Trimmed whitespace and handled empty required text",
            "Combined short_description and description into model_text",
            "Excluded rows with empty text or missing root_cause from supervised training",
            "Excluded identical text appearing with conflicting root-cause labels",
            "Removed exact duplicate model_text rows",
            "Vectorized text using a TextVectorization layer adapted only on training text",
        ],
        "note": (
            "Unlabelled rows are excluded from supervised classifier training. "
            "This does not mean those incidents are useless for a later grouping-only component."
        ),
    })
    PREPROCESSING_PATH.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return df, summary


def split_data(df, y):
    """
    Prefer a group-aware split if incident_group_id is present and useful.
    This keeps records from the same generated/event group together.
    Otherwise, use a stratified random split.
    """
    group_col = next(
        (c for c in ["incident_group_id", "group_id", "event_group_id"] if c in df.columns),
        None,
    )
    split_note = ""
    if group_col:
        groups = df[group_col].fillna("").astype(str).str.strip()
        valid_group_mask = groups.ne("") & groups.ne("nan") & groups.ne("None")
        # Group split only when enough non-empty groups exist.
        if valid_group_mask.all() and groups.nunique() >= 5:
            try:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
                train_val_idx, test_idx = next(
                    splitter.split(np.zeros(len(df)), y, groups=groups)
                )
                # Split train/validation by group too, so no group appears in both.
                second_splitter = GroupShuffleSplit(
                    n_splits=1, test_size=0.20, random_state=SEED + 1
                )
                train_local, val_local = next(
                    second_splitter.split(
                        np.zeros(len(train_val_idx)),
                        y[train_val_idx],
                        groups=groups.iloc[train_val_idx].to_numpy(),
                    )
                )
                train_idx = train_val_idx[train_local]
                val_idx = train_val_idx[val_local]
                # Check that training includes every class; if not, try stratified fallback.
                if len(np.unique(y[train_idx])) >= len(np.unique(y)) and len(test_idx) > 0:
                    split_note = (
                        f"Group-aware split using '{group_col}': approximately 64% train, "
                        "16% validation, 20% test. Groups are kept separate across all splits."
                    )
                    return train_idx, val_idx, test_idx, split_note, group_col
                split_note = "Group split did not preserve all classes in training; used stratified fallback."
            except Exception as exc:
                split_note = f"Group split unavailable ({type(exc).__name__}); used stratified fallback."

    all_idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.20, random_state=SEED, stratify=y[train_val_idx]
    )
    fallback_note = (
        "Stratified random split: 64% train, 16% validation, 20% test. "
        "If repeated event/group IDs exist, this fallback may permit group leakage."
    )
    if split_note:
        fallback_note += " " + split_note
    return train_idx, val_idx, test_idx, fallback_note, None


def save_training_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.history.get("accuracy", []), label="Training accuracy")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation accuracy")
    axes[0].set_title("BiLSTM Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(history.history.get("loss", []), label="Training loss")
    axes[1].plot(history.history.get("val_loss", []), label="Validation loss")
    axes[1].set_title("BiLSTM Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(HISTORY_PNG_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 72)
    print("DEEP LEARNING-BASED COMMON-CAUSE DISCOVERY IN ITSM")
    print("=" * 72)

    df, prep_summary = load_and_preprocess()
    print("\nPREPROCESSING SUMMARY")
    print(f"Dataset: {DATA_PATH.name}")
    print(f"Rows at start: {prep_summary['rows_at_start']}")
    print(f"Rows used for supervised training: {len(df)}")
    print(f"Rows without root-cause labels excluded: "
          f"{prep_summary['rows_without_root_cause_label_excluded_from_supervised_training']}")
    print(f"Conflicting text rows excluded: {prep_summary['conflicting_text_rows_excluded']}")
    print(f"Exact duplicate text rows removed: {prep_summary['exact_duplicate_text_rows_removed']}")
    print("\nClass counts:")
    print(df["root_cause"].value_counts().sort_index().to_string())

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["root_cause"].astype(str))
    X_all = df["model_text"].astype(str).to_numpy()

    train_idx, val_idx, test_idx, split_note, group_col = split_data(df, y_all)
    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    # Vectorizer learns its vocabulary from training data only.
    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=TEXT_SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(64))

    n_classes = len(label_encoder.classes_)
    inputs = layers.Input(shape=(), dtype=tf.string, name="incident_text")
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBEDDING_DIM, mask_zero=True, name="embedding")(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.20, recurrent_dropout=0.0))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="root_cause_probabilities")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="ITSM_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights help if some root-cause classes are less common.
    train_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=train_classes, y=y_train
    )
    class_weight = {int(c): float(w) for c, w in zip(train_classes, weights)}

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )

    print("\nDATA SPLIT")
    print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")
    print(split_note)
    print("\nMODEL SUMMARY")
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=1,
    )
    save_training_plot(history)

    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_examples": int(len(train_idx)),
        "validation_examples": int(len(val_idx)),
        "test_examples": int(len(test_idx)),
        "unique_labelled_incidents_used": int(len(df)),
        "number_of_classes": int(n_classes),
        "class_names": [str(x) for x in label_encoder.classes_],
        "split_method": split_note,
        "group_column_used": group_col,
        "epochs_trained": int(len(history.history.get("loss", []))),
        "best_validation_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "caution": (
            "Metrics are for this dataset split only. Synthetic/template similarity can inflate results. "
            "A separate external dataset and group-discovery evaluation are needed before claiming "
            "real-world common-cause discovery."
        ),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=[str(x) for x in label_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(REPORT_PATH, index=True)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {x}" for x in label_encoder.classes_],
        columns=[f"Predicted: {x}" for x in label_encoder.classes_],
    )
    cm_df.to_csv(CM_CSV_PATH)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, xticks_rotation=70, colorbar=False, values_format="d")
    ax.set_title("BiLSTM Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CM_PNG_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)

    test_rows = df.iloc[test_idx].copy()
    test_rows["actual_root_cause"] = label_encoder.inverse_transform(y_test)
    test_rows["predicted_root_cause"] = label_encoder.inverse_transform(y_pred)
    test_rows["prediction_confidence"] = np.max(probabilities, axis=1)
    test_rows["correct_prediction"] = test_rows["actual_root_cause"] == test_rows["predicted_root_cause"]
    test_rows.to_csv(PREDICTIONS_PATH, index=False)

    # Candidate groups are a convenience report, NOT causal proof.
    # Group test incidents with the same predicted class and minimum confidence.
    group_rows = test_rows[test_rows["prediction_confidence"] >= 0.60].copy()
    if len(group_rows):
        group_rows["candidate_group_size"] = group_rows.groupby("predicted_root_cause")[
            "predicted_root_cause"
        ].transform("size")
        group_rows = group_rows[group_rows["candidate_group_size"] >= 2].copy()
        keep_cols = [
            c for c in [
                "short_description", "description", "predicted_root_cause",
                "prediction_confidence", "candidate_group_size", "actual_root_cause",
                "correct_prediction", "incident_group_id", "source", "is_synthetic"
            ] if c in group_rows.columns
        ]
        group_rows[keep_cols].sort_values(
            ["predicted_root_cause", "prediction_confidence"], ascending=[True, False]
        ).to_csv(GROUPS_PATH, index=False)
    else:
        pd.DataFrame(columns=[
            "short_description", "description", "predicted_root_cause",
            "prediction_confidence", "candidate_group_size"
        ]).to_csv(GROUPS_PATH, index=False)

    # LIME explains one held-out prediction locally; it is not a causal explanation.
    lime_status = "not generated"
    try:
        from lime.lime_text import LimeTextExplainer

        class_names = [str(x) for x in label_encoder.classes_]

        def predict_proba_for_lime(texts):
            texts = np.asarray(texts, dtype=str)
            return model.predict(texts, batch_size=64, verbose=0)

        explainer = LimeTextExplainer(class_names=class_names)
        explanation = explainer.explain_instance(
            X_test[0],
            predict_proba_for_lime,
            num_features=10,
            num_samples=1000,
        )
        explanation.save_to_file(str(LIME_PATH))
        lime_status = str(LIME_PATH.name)
    except Exception as exc:
        print(f"\nLIME explanation could not be generated: {exc}")

    model.save(MODEL_PATH)
    metadata = {
        "class_names": [str(x) for x in label_encoder.classes_],
        "max_tokens": MAX_TOKENS,
        "sequence_length": TEXT_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "architecture": "TextVectorization -> Embedding -> Bidirectional LSTM -> Dense softmax",
        "dataset_file": DATA_PATH.name,
        "text_column": "model_text",
        "split_method": split_note,
        "group_column_used": group_col,
        "lime_file": lime_status,
    }
    joblib.dump(metadata, METADATA_PATH)

    prep_summary["split_method"] = split_note
    prep_summary["train_rows"] = int(len(train_idx))
    prep_summary["validation_rows"] = int(len(val_idx))
    prep_summary["test_rows"] = int(len(test_idx))
    PREPROCESSING_PATH.write_text(json.dumps(json_safe(prep_summary), indent=2), encoding="utf-8")
    EVALUATION_PATH.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FINAL TEST METRICS")
    print("=" * 72)
    for key in [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted"
    ]:
        print(f"{key.replace('_', ' ').title():25s}: {metrics[key]:.4f}")
    print(f"\nDetailed report saved to: {REPORT_PATH.name}")
    print(f"Confusion matrix saved to: {CM_PNG_PATH.name} and {CM_CSV_PATH.name}")
    print(f"Training curves saved to: {HISTORY_PNG_PATH.name}")
    print(f"Preprocessing summary saved to: {PREPROCESSING_PATH.name}")
    print(f"Evaluation JSON saved to: {EVALUATION_PATH.name}")
    print(f"Test predictions saved to: {PREDICTIONS_PATH.name}")
    print(f"Candidate groups saved to: {GROUPS_PATH.name}")
    print(f"LIME explanation: {lime_status}")
    print(f"Model saved to: {MODEL_PATH.name}")
    print("\nIMPORTANT: Candidate groups are suggestions based on predicted labels, not verified causal links.")


if __name__ == "__main__":
    main()
