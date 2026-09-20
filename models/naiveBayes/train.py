from dataset_loader import load_dataset
from preprocess import clean_text

from models.naiveBayes.naive_bayes_sam import (
    NaiveBayesSAM
)

from models.naiveBayes.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH
)


def main():

    data = load_dataset()

    data["clean_review"] = (
        data["review"].apply(clean_text)
    )

    sam = NaiveBayesSAM()

    sam.train(
        data["clean_review"],
        data["sentiment"]
    )

    sam.save(MODEL_PATH)

    print("Naive Bayes model saved")


if __name__ == "__main__":
    main()