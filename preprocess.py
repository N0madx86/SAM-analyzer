import re


RATING_POS = "<rating_pos>"
RATING_NEG = "<rating_neg>"
NUM = "<num>"

CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
}

def normalize_contractions(text):

    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)

    return text


def remove_html(text):
    # clean ALL HTML tags
    return re.sub(r"<[^>]+>", " ", text)


def normalize_case(text):
    return text.lower()


def normalize_ratings(text):

    # -------- 10-point ratings --------
    # 0-5 /10 -> negative rating
    text = re.sub(r"\b([6-9](\.\d+)?|10(\.0+)?)/10\b", f"{RATING_NEG}", text)
    # 6-10 /10 -> positive rating
    text = re.sub(r"\b([0-5](\.\d+)?)/10\b", f"{RATING_POS}", text)

    # -------- 5-star ratings --------
    # 1-2 stars -> Negative
    text = re.sub(r"\b([1-2])(\.\d+)?\s*(stars|star)\b", f"{RATING_NEG}", text)
    # 3-5 stars -> Positive
    text = re.sub(r"\b([3-5])(\.\d+)?\s*(stars|star)\b", f"{RATING_POS}", text)

    # -------- x/5 ratings --------
    # 1-2/5 -> Negative
    text = re.sub(r"\b([1-2](\.\d+)?)/5\b", f"{RATING_NEG}", text)
    # 3-5/5 -> Positive
    text = re.sub(r"\b([3-5](\.\d+)?)/5\b", f"{RATING_POS}", text)

    return text


def normalize_numbers(text):
    # Replaces remaining standalone numbers that weren't part of ratings
    return re.sub(r"\b\d+(\.\d+)?\b", f"{NUM}", text)


def remove_punctuation(text):
    # FIXED: Added '_' to the allowed characters to preserve your tokens
    return re.sub(r"[^a-z0-9<>_\s]", " ", text)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    #Clean structural noise first
    text = remove_html(text)
    text = normalize_case(text)
    text = normalize_contractions(text)

    #Extract features from specific to general (Ratings before standalone numbers)
    text = normalize_ratings(text)
    text = normalize_numbers(text)

    #Strip punctuation while preserving the custom tokens
    text = remove_punctuation(text)

    #Collapse spaces left behind by punctuation removal
    text = normalize_whitespace(text)

    return text


if __name__ == "__main__":
    test = """
        This movie was 10/10!!!
        I watched it 3 times in 2024.<br /><br />
        Definitely 5 stars.
        """
    print(clean_text(test))
