from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import feedparser
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Sentence embedding model (CPU-friendly)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Reddit Sentiment & Summary Analyzer")


class SubredditRequest(BaseModel):
    subreddit: str
    top_n: int = 25


def query_mistral(prompt: str, max_tokens: int = 150) -> str:
    """
    Query the running Ollama Mistral model via HTTP (non-streaming).
    Returns the full text response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "max_tokens": max_tokens
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        return f"Error: {response.status_code} - {response.text}"


# Sentiment analysis model (CPU-friendly)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)


@app.post("/analyze")
def analyze_subreddit(req: SubredditRequest):
    # Validate subreddit URL
    if " " in req.subreddit or not req.subreddit.strip():
        raise HTTPException(status_code=400, detail="Invalid subreddit name or URL")

    rss_url = f"https://www.reddit.com/r/{req.subreddit}/.rss"
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise HTTPException(status_code=404, detail="Subreddit not found or RSS feed empty")

    entries = feed.entries[:req.top_n]
    titles = [entry.title for entry in entries]

    # Sentiment analysis
    sentiments = [sentiment_analyzer(title)[0]['label'] for title in titles]
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for s in sentiments:
        if s == "POSITIVE":
            counts["POSITIVE"] += 1
        elif s == "NEGATIVE":
            counts["NEGATIVE"] += 1
        else:
            counts["NEUTRAL"] += 1

    # Plot sentiment distribution in memory
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=["green", "red", "gray"])
    plt.title(f"Sentiment Analysis of /r/{req.subreddit} (last {req.top_n} posts)")
    plt.ylabel("Number of posts")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Mistral summary (non-streaming)
    posts_text = " ".join(titles)
    prompt = f"Summarize the following Reddit posts in 3 concise sentences:\n{posts_text}."
    summary = query_mistral(prompt, max_tokens=150)

    return JSONResponse(content={
        "summary": summary,
        "sentiment_graph": img_base64,
        #"counts": counts
    })
