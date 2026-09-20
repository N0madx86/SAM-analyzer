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
    # Train / Test Split
    # ======================================================

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples    : {len(x_test)}")

    # ======================================================
    # Model
    # ======================================================

    sam = LogisticSAM()

    sam.train(
        x_train,
        y_train
    )

    # ======================================================
    # Save
    # ======================================================

    sam.save(
        MODEL_PATH
    )

    print("Logistic Regression model saved")


if __name__ == "__main__":
    main()