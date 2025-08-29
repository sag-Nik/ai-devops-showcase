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
import os
from fastapi.middleware.cors import CORSMiddleware


FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5174")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Sentence embedding model (CPU-friendly) for future improvements
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Reddit Sentiment & Summary Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SubredditRequest(BaseModel):
    subreddit: str
    top_n: int = 25
    temperature: float = 0.5  
    max_tokens: int = 150     


def query_mistral(prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
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

    filtered_counts = {k: v for k, v in counts.items() if v > 0}

    if filtered_counts:
        colors_map = {"POSITIVE": "#34D399", "NEGATIVE": "#F87171", "NEUTRAL": "#FBBF24"}
        colors = [colors_map[k] for k in filtered_counts.keys()]
        explode = tuple(0.05 for _ in filtered_counts)  # slight separation

        plt.figure(figsize=(6,6))
        plt.pie(
            filtered_counts.values(),
            labels=filtered_counts.keys(),
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=explode,
            shadow=True,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        plt.title(f"Sentiment Analysis of /r/{req.subreddit} (last {req.top_n} posts)",
                fontsize=14, fontweight='bold', color='#111827')

        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='white')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    else:
        img_base64 = None  # or handle empty data gracefully



    # Mistral summary
    posts_text = " ".join(titles)
    prompt = f"IN exactly 3 short sentences summarise the overall sentiment of the following reddit posts :\n{posts_text}."
    summary = query_mistral(
        prompt, 
        max_tokens=req.max_tokens, 
        temperature=req.temperature
    )

    return JSONResponse(content={
        "summary": summary,
        "sentiment_graph": img_base64,
    })