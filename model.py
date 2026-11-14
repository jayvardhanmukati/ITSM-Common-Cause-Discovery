
"""
ITSM Common-Cause Discovery
Deep Learning Model: Bidirectional LSTM + LIME Explainability

Input dataset: incidents.csv
Required columns: short_description, description, root_cause

This script creates a separate deep-learning model.
It does NOT modify app.py or model_bundle.joblib.
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from tensorflow.keras import layers, callbacks, Model
from lime.lime_text import LimeTextExplainer


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42
DATA_PATH = Path("incidents.csv")

MODEL_PATH = Path("lstm_model.keras")
METADATA_PATH = Path("lstm_metadata.joblib")
RESULTS_PATH = Path("lstm_evaluation.json")
LIME_PATH = Path("lstm_lime_explanation.html")

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 120
EMBEDDING_DIM = 64
LSTM_UNITS = 32
BATCH_SIZE = 16
EPOCHS = 15

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH.resolve()}. "
            "Run this script from your project folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "short_description",
        "description",
        "root_cause",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Remove rows that have no usable text or target label.
    df = df.dropna(
        subset=["short_description", "description", "root_cause"]
    ).copy()

    df["root_cause"] = df["root_cause"].astype(str).str.strip()
    df["short_description"] = (
        df["short_description"].astype(str).str.strip()
    )
    df["description"] = df["description"].astype(str).str.strip()

    df["model_text"] = (
        df["short_description"] + ". " + df["description"]
    ).str.strip()

    df = df[
        (df["model_text"] != "")
        & (df["root_cause"] != "")
    ].copy()

    # Remove exact duplicate incident texts to reduce data leakage.
    df = df.drop_duplicates(subset=["model_text"]).reset_index(drop=True)

    if df["root_cause"].nunique() < 2:
        raise ValueError(
            "At least two different root-cause categories are required."
        )

    class_counts = df["root_cause"].value_counts()

    if (class_counts < 2).any():
        rare_classes = class_counts[class_counts < 2].to_dict()
        raise ValueError(
            "Each root-cause category needs at least two unique "
            f"incident texts for a stratified split. Problem categories: "
            f"{rare_classes}"
        )

    print("\n" + "=" * 60)
    print("ITSM INCIDENT DATASET")
    print("=" * 60)
    print(f"Unique incident texts: {len(df)}")
    print(f"Root-cause categories: {df['root_cause'].nunique()}")
    print("\nSamples per category:")
    print(class_counts.to_string())

    return df


# --------------------------------------------------
# 3. SPLIT DATA AND ENCODE LABELS
# --------------------------------------------------

def prepare_data(df):
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["root_cause"])
    X = df["model_text"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Training incidents: {len(X_train)}")
    print(f"Testing incidents:  {len(X_test)}")
    print(f"Number of classes:  {len(label_encoder.classes_)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
    )


# --------------------------------------------------
# 4. BUILD THE LSTM DEEP LEARNING MODEL
# --------------------------------------------------

def build_model(X_train, number_of_classes):
    # Convert text into integer token sequences.
    # Adapt only on training text to avoid test-data leakage.
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        name="text_vectorization",
    )

    train_text_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    vectorizer.adapt(train_text_dataset.batch(BATCH_SIZE))

    actual_vocab_size = len(vectorizer.get_vocabulary())

    # Model accepts raw incident text as input.
    text_input = layers.Input(
        shape=(),
        dtype=tf.string,
        name="incident_text",
    )

    x = vectorizer(text_input)

    # Learn a vector representation for each token.
    x = layers.Embedding(
        input_dim=actual_vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="word_embedding",
    )(x)

    # Bidirectional LSTM learns patterns across incident text.
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS),
        name="bidirectional_lstm",
    )(x)

    x = layers.Dropout(0.35, name="dropout")(x)
    x = layers.Dense(32, activation="relu", name="dense_features")(x)

    output = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="root_cause_probabilities",
    )(x)

    model = Model(
        inputs=text_input,
        outputs=output,
        name="ITSM_Bidirectional_LSTM",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n" + "=" * 60)
    print("LSTM MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()

    print(f"\nVocabulary size: {actual_vocab_size}")

    return model


# --------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------

def train_model(model, X_train, y_train):
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("TRAINING THE DEEP LEARNING MODEL")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=True,
    )

    return history


# --------------------------------------------------
# 6. EVALUATE ON HELD-OUT TEST DATA
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, label_encoder):
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    class_names = label_encoder.classes_.tolist()
    class_ids = list(range(len(class_names)))

    report = classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_ids,
    )

    print("\n" + "=" * 60)
    print("HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"Test accuracy:       {accuracy:.4f} ({accuracy:.2%})")
    print(f"Weighted F1-score:   {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_string())

    results = {
        "model": "Bidirectional LSTM",
        "dataset": str(DATA_PATH),
        "unique_incidents": int(len(X_test) + len(y_test)),
        "test_samples": int(len(y_test)),
        "number_of_classes": int(len(class_names)),
        "test_accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nEvaluation results saved to: {RESULTS_PATH.resolve()}")

    return predictions, probabilities


# --------------------------------------------------
# 7. LIME EXPLAINABILITY
# --------------------------------------------------

def explain_prediction(model, X_test, probabilities, label_encoder):
    print("\n" + "=" * 60)
    print("LIME EXPLAINABILITY")
    print("=" * 60)

    # Use the first held-out test incident as an example.
    sample_text = str(X_test[0])
    sample_probabilities = probabilities[0]

    predicted_index = int(np.argmax(sample_probabilities))
    predicted_cause = label_encoder.inverse_transform(
        [predicted_index]
    )[0]
    confidence = float(sample_probabilities[predicted_index])

    print("\nExample incident:")
    print(sample_text)
    print(f"\nPredicted root cause: {predicted_cause}")
    print(f"Model confidence: {confidence:.2%}")
    print("Note: confidence is not proof that the cause is correct.")

    def predict_probabilities(texts):
        # LIME generates perturbed versions of the input text.
        texts = np.asarray(texts, dtype=str)
        return model.predict(texts, verbose=0)

    explainer = LimeTextExplainer(
        class_names=label_encoder.classes_.tolist(),
        random_state=SEED,
    )

    explanation = explainer.explain_instance(
        sample_text,
        predict_probabilities,
        labels=[predicted_index],
        num_features=10,
        num_samples=1000,
    )

    print("\nWords influencing the predicted category:")
    for word, weight in explanation.as_list(label=predicted_index):
        direction = "supports" if weight > 0 else "opposes"
        print(f"  {word}: {weight:+.4f} ({direction} this prediction)")

    explanation.save_to_file(str(LIME_PATH))

    print(f"\nInteractive LIME explanation saved to: {LIME_PATH.resolve()}")
    print("Open this HTML file in a browser to inspect the explanation.")

    return predicted_cause, confidence


# --------------------------------------------------
# 8. SAVE MODEL AND METADATA
# --------------------------------------------------

def save_artifacts(model, label_encoder):
    # Save the complete model, including its text vectorization layer.
    model.save(MODEL_PATH)

    # Keep label mapping separately for future inference and dashboard use.
    metadata = {
        "class_names": label_encoder.classes_.tolist(),
        "model_type": "Bidirectional LSTM",
        "text_column": "short_description + description",
        "target_column": "root_cause",
        "sequence_length": SEQUENCE_LENGTH,
        "vocabulary_size": VOCAB_SIZE,
    }

    joblib.dump(metadata, METADATA_PATH)

    print("\n" + "=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Trained model: {MODEL_PATH.resolve()}")
    print(f"Metadata:      {METADATA_PATH.resolve()}")


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

def main():
    print("\nITSM COMMON-CAUSE DISCOVERY")
    print("Deep Learning Model: LSTM + LIME")

    df = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
    ) = prepare_data(df)

    model = build_model(
        X_train,
        number_of_classes=len(label_encoder.classes_),
    )

    train_model(model, X_train, y_train)

    predictions, probabilities = evaluate_model(
        model,
        X_test,
        y_test,
        label_encoder,
    )

    save_artifacts(model, label_encoder)

    explain_prediction(
        model,
        X_test,
        probabilities,
        label_encoder,
    )

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED")
    print("=" * 60)
    print("Your existing app.py and model_bundle.joblib were not modified.")
    print("Review the evaluation results before relying on the predictions.")


if __name__ == "__main__":
    main()
