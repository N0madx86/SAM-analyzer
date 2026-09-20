from dataset_loader import load_dataset
from preprocess import clean_text

from models.logisticRegression.logistic_reression_sam import (
    LogisticSAM
)

from models.logisticRegression.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH
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


def main():

    # ======================================================
    # Dataset
    # ======================================================

    data = load_dataset()

    data["clean_review"] = (
        data["review"].apply(clean_text)
    )

    texts = data["clean_review"].tolist()
    labels = data["sentiment"].tolist()

    # ======================================================
    # Recreate Exact Test Split
    # ======================================================

    _, x_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ======================================================
    # Load Trained Model
    # ======================================================

    sam = LogisticSAM()

    sam.load(
        MODEL_PATH
    )

    print("Trained model loaded")

    # ======================================================
    # Prediction
    # ======================================================

    predictions = []

    for text in x_test:

        predictions.append(
            sam.predict(text)
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