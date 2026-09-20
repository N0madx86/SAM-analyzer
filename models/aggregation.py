class SentimentAggregator:

    def aggregate(self, predictions):

        sentiments = [
            result["sentiment"].lower()
            for result in predictions.values()
        ]

        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")

        total = len(sentiments)

        positive_ratio = (
            positive_count / total
        )

        negative_ratio = (
            negative_count / total
        )

        net_ratio = (
            positive_ratio -
            negative_ratio
        )

        if net_ratio > 0:
            final_sentiment = "positive"

        elif net_ratio < 0:
            final_sentiment = "negative"

        else:
            final_sentiment = "neutral"

        return {
            "sentiment": final_sentiment,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "net_ratio": net_ratio,
            "model_count": total
        }