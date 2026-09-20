from models.vader.vader_sam import *

def main(sam):
    while True:
        text_in = input("enter text: ")
        if (text_in.lower() == "quit"):
            sam.print_history()
            break
        sam.analyze_sentiment(text_in)



if __name__ == "__main__":
    sam = SAM()
    main(sam)
