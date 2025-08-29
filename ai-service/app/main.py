from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import feedparser
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Sentence embedding model (CPU-friendly)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Reddit Sentiment & Summary Analyzer")


class SubredditRequest(BaseModel):
    subreddit: str
    top_n: int = 25


def query_mistral_stream(prompt: str, max_tokens: int = 150):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.5,
            "max_tokens": max_tokens
        }
    }
    headers = {"Content-Type": "application/json"}
    buffer = ""
    with requests.post(OLLAMA_URL, json=payload, headers=headers, stream=True) as response:
        if response.status_code != 200:
            yield f"Error: {response.status_code} - {response.text}"
            return
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    buffer += chunk
                    # split on punctuation and yield completed sentences
                    while "." in buffer:
                        sentence, buffer = buffer.split(".", 1)
                        yield sentence.strip() + ".\n"
                except:
                    continue
        # yield any remaining text
        if buffer.strip():
            yield buffer.strip()



sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

@app.post("/analyze")
def analyze_subreddit_stream(req: SubredditRequest):
    rss_url = f"https://www.reddit.com/r/{req.subreddit}/.rss"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise HTTPException(status_code=404, detail="Subreddit not found or RSS feed empty")

    entries = feed.entries[:req.top_n]
    titles = [entry.title for entry in entries]

    # Sentiment analysis (same as before)
    sentiments = [sentiment_analyzer(title)[0]['label'] for title in titles]
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for s in sentiments:
        if s == "POSITIVE":
            counts["POSITIVE"] += 1
        elif s == "NEGATIVE":
            counts["NEGATIVE"] += 1
        else:
            counts["NEUTRAL"] += 1

    # Plot sentiment distribution
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=["green", "red", "gray"])
    plt.title(f"Sentiment Analysis of /r/{req.subreddit} (last {req.top_n} posts)")
    plt.ylabel("Number of posts")
    plt.tight_layout()
    graph_file = f"{req.subreddit}_sentiment.png"
    plt.savefig(graph_file)
    plt.close()

    # Mistral streaming
    posts_text = " ".join(titles)
    prompt = f"Summarize the following Reddit posts in 3 concise sentences:\n{posts_text}. Dont respond with the summary, just speak about the overall sentiment."

    return StreamingResponse(query_mistral_stream(prompt, max_tokens=150), media_type="text/plain")
