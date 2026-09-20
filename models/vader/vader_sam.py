from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SAM:
    def __init__(self):

        self.analyzer = SentimentIntensityAnalyzer()
        self.history = []

    def get_sentiment_scores(self,text):

        return self.analyzer.polarity_scores(text)

    def determine_sentiment(self,compound):

        if (compound > 0.05):
            return "positive"
        elif (compound < -0.05):
            return "negative"
        else:
            return "neutral"




    def update_history(self,text_in,sentiment,compound_score):
        self.history.append({"text": text_in,
                             "sentiment": sentiment,
                             "score": compound_score})





    def analyze_sentiment(self,text_in):

        result = self.get_sentiment_scores(text_in)
        compound_score = result["compound"]

        sentiment = self.determine_sentiment(compound_score)

        self.update_history(text_in,sentiment,compound_score)

        self.display_results(text_in, result, compound_score, sentiment)





    def display_results(self, text, result, compound, sentiment):

        print("\n----- Analysis -----")
        print(f"Text: {text}")
        print(f"Positive Score: {result['pos']:.2f}")
        print(f"Negative Score: {result['neg']:.2f}")
        print(f"Neutral Score: {result['neu']:.2f}")
        print(f"Compound Score: {compound:.2f}")
        print(f"Final Sentiment: {sentiment}")





    def print_history(self):

        print("----- History -----")
        for item in self.history:
            print(f"\ntext: {item['text']}\n"
                  f"sentiment: {item['sentiment']}\n"
                  f"score: {item['score']:.2f}\n"
                  f"-------------------")

