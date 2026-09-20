# ==========================================================
# Imports
# ==========================================================

import torch
from torch.utils.data import DataLoader

from dataset_loader import load_dataset
from preprocess import clean_text

from models.lstm.dataset import ReviewData
from models.lstm.lstm_sam import LSTMSAM
from models.lstm.tokenizer import Tokenizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from models.lstm.config import (
    BATCH_SIZE,
    MAX_FEATURES,
    MIN_FREQ,
    MAX_LENGTH,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH
)



# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ==========================================================
# Dataset Loading & Preprocessing
# ==========================================================

data = load_dataset()

print("Dataset loaded")

data["clean_review"] = (
    data["review"].apply(clean_text)
)

data["label"] = data["sentiment"].map({
    "negative": 0,
    "positive": 1
})


# ==========================================================
# Tokenization
# ==========================================================

tokenizer = Tokenizer(
    max_features=MAX_FEATURES,
    min_freq=MIN_FREQ,
    max_length=MAX_LENGTH
)

tokenizer.build_vocab(
    data["clean_review"]
)

data["sequence"] = data["clean_review"].apply(
    tokenizer.text_to_sequence
)

padded_sequences = tokenizer.pad_sequences(
    data["sequence"].tolist()
)


# ==========================================================
# Same Train / Validation / Test Split
# ==========================================================

x_train, x_temp, y_train, y_temp = train_test_split(
    padded_sequences,
    data["label"].tolist(),
    test_size=0.30,
    random_state=RANDOM_STATE
)

x_validation, x_test, y_validation, y_test = (
    train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )
)


# ==========================================================
# Test Tensors
# ==========================================================

x_test = torch.tensor(
    x_test,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# ==========================================================
# Test Dataset & DataLoader
# ==========================================================

test_dataset = ReviewData(
    x_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================================
# Model
# ==========================================================

model = LSTMSAM(
    vocab_size=len(tokenizer.word_to_index)
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

print("Best checkpoint loaded")


# ==========================================================
# Evaluation
# ==========================================================

print("\nTesting Started\n")

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for x_batch, y_batch in test_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            predicted.cpu().tolist()
        )

        actuals.extend(
            y_batch.cpu().tolist()
        )


# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    actuals,
    predictions
)

precision = precision_score(
    actuals,
    predictions
)

recall = recall_score(
    actuals,
    predictions
)

f1 = f1_score(
    actuals,
    predictions
)

matrix = confusion_matrix(
    actuals,
    predictions
)


# ==========================================================
# Results
# ==========================================================

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")

print(
    classification_report(
        actuals,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ]
    )
)

print("\nConfusion Matrix")

print(matrix)