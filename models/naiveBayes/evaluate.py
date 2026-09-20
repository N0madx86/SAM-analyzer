from dataset_loader import load_dataset
from preprocess import clean_text

from models.naiveBayes.naive_bayes_sam import (
    NaiveBayesSAM
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.naiveBayes.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH
)


def main():

    # ======================================================
    # Dataset
    # ======================================================

    data = load_dataset()

    data["clean_review"] = (
        data["review"].apply(clean_text)
    )

    labels = data["sentiment"].tolist()

    # ======================================================
    # Load trained model
    # ======================================================

    sam = NaiveBayesSAM()

    sam.load(MODEL_PATH)

    # ======================================================
    # Recreate exact test split
    # ======================================================

    _, x_test, _, y_test = train_test_split(
        data["clean_review"].tolist(),
        labels,
        test_size=0.3,
        random_state=42
    )

    # ======================================================
    # Transform test data
    # ======================================================

    x_test_vectors = sam.vectorizer.transform(
        x_test
    )

    # ======================================================
    # Predict
    # ======================================================

    predictions = sam.model.predict(
        x_test_vectors
    )

    # ======================================================
    # Metrics
    # ======================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="positive"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="positive"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="positive"
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # ======================================================
    # Results
    # ======================================================

    print("\nTest Results")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Negative",
                "Positive"
            ]
        )
    )

    print("\nConfusion Matrix")

    print(matrix)


if __name__ == "__main__":
    main()