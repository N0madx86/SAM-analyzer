import pandas as pd

def load_dataset():
    data = pd.read_csv("D:\\Projects\\SAM\\data\\IMDB Dataset.csv")
    return data


if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    print(data.shape)
    print(data["sentiment"].value_counts())
