from models.inference_engine import SentimentEngine


def main():

    engine = SentimentEngine()

    text = "This product is absolutely amazing and I love it."

    result = engine.predict(text)

    print("\nUnified Result:")

    for model, prediction in result.items():
        print(f"{model}: {prediction}")


if __name__ == "__main__":
    main()