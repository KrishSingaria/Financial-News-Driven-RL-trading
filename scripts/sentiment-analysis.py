import config
import json
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification

def daterange(start_date, end_date):
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)

def analyze_sentiment(texts, tokenizer, model):
    if not texts:
        return 0.0
    scores = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=1)[0]
            score = (-1 * probs[0] + 0 * probs[1] + 1 * probs[2]).item()
            scores.append(score)
    return sum(scores) / len(scores)

def main():
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME)
    model.eval()

    config.SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end = datetime.strptime(config.END_DATE, "%Y-%m-%d")

    for ticker in config.TICKERS:
        print(f"Processing sentiment for {ticker}")
        for single_date in daterange(start, end):
            date_str = single_date.strftime("%Y-%m-%d")
            news_path = config.NEWS_DIR / ticker / f"{date_str}.json"
            out_path = config.SENTIMENT_DIR / ticker
            out_path.mkdir(parents=True, exist_ok=True)
            save_file = out_path / f"{date_str}.json"

            if save_file.exists() or not news_path.exists():
                continue

            try:
                with open(news_path, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                headlines = [a["headline"] for a in articles if "headline" in a]
                score = analyze_sentiment(headlines, tokenizer, model)

                with open(save_file, "w", encoding="utf-8") as out:
                    json.dump({"date": date_str, "score": score}, out)

                print(f"{ticker} | {date_str} → Sentiment: {score:.3f}")
            except Exception as e:
                print(f"[ERROR] {ticker} {date_str}: {e}")

if __name__ == "__main__":
    main()
