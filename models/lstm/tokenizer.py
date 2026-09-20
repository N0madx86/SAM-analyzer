from collections import Counter


class Tokenizer:

    def __init__(
        self,
        max_features=50000,
        min_freq=2,
        max_length=200
    ):

        self.max_features = max_features
        self.min_freq = min_freq
        self.max_length = max_length

        self.word_to_index = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.index_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build_vocab(self, texts):

        # Count word frequencies
        word_counts = Counter()

        for text in texts:
            word_counts.update(text.split())

        # Remove rare words
        filtered_words = {
            word: count
            for word, count in word_counts.items()
            if count >= self.min_freq
        }

        # Sort by:
        # 1. Frequency (descending)
        # 2. Alphabetically (ascending)
        sorted_words = sorted(
            filtered_words.items(),
            key=lambda x: (-x[1], x[0])
        )

        # Limit vocabulary size
        if self.max_features is not None:
            sorted_words = sorted_words[:self.max_features - 2]

        vocab = [
            word
            for word, _
            in sorted_words
        ]

        # Build lookup tables
        for idx, word in enumerate(vocab, start=2):

            self.word_to_index[word] = idx
            self.index_to_word[idx] = word

        print(f"Vocabulary Size : {len(self.word_to_index)}")
        print(f"Minimum Frequency : {self.min_freq}")
        print(f"Maximum Features : {self.max_features}")

    def text_to_sequence(self, text):

        return [
            self.word_to_index.get(
                token,
                self.word_to_index["<UNK>"]
            )
            for token in text.split()
        ]

    def pad_sequences(self, seqs):

        padded_sequences = []

        pad_id = self.word_to_index["<PAD>"]

        for seq in seqs:

            # Truncate long reviews
            seq = seq[:self.max_length]

            # Pad short reviews
            seq = seq + (
                [pad_id] *
                (self.max_length - len(seq))
            )

            padded_sequences.append(seq)

        return padded_sequences

    def save_vocab(self, path):

        import pickle

        with open(path, "wb") as file:
            pickle.dump(
                self.word_to_index,
                file
            )

    def load_vocab(self, path):

        import pickle

        with open(path, "rb") as file:
            self.word_to_index = pickle.load(file)