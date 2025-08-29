from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import feedparser
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
import os
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Environment & Config
# -----------------------------
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5174")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "mistral"

# -----------------------------
# Initialize Models
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU-friendly embeddings

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(title="Reddit Sentiment & Summary Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class SubredditRequest(BaseModel):
    subreddit: str
    top_n: int = 25
    temperature: float = 0.5
    max_tokens: int = 150

# -----------------------------
# Helper Functions
# -----------------------------
def query_mistral(prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
    """
    Query the Ollama Mistral model via HTTP and return the full response.
    """
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
    return f"Error: {response.status_code} - {response.text}"


def generate_sentiment_chart(counts: dict[str, int], subreddit: str, top_n: int) -> str | None:
    """
    Generate a pie chart for sentiment distribution and return as base64 string.
    Excludes zero-count sentiments.
    """
    filtered_counts = {k: v for k, v in counts.items() if v > 0}
    if not filtered_counts:
        return None

    colors_map = {"POSITIVE": "#15FF00", "NEGATIVE": "#ff4500", "NEUTRAL": "#FBBF24"}
    colors = [colors_map[k] for k in filtered_counts.keys()]
    explode = tuple(0.05 for _ in filtered_counts)

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
    plt.title(
        f"Sentiment Analysis of /r/{subreddit} (last {top_n} posts)",
        fontsize=14,
        fontweight='bold',
        color='#111827'
    )
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/analyze")
def analyze_subreddit(req: SubredditRequest) -> JSONResponse:
    # Validate subreddit
    subreddit_clean = req.subreddit.strip()
    if not subreddit_clean or " " in subreddit_clean:
        raise HTTPException(status_code=400, detail="Invalid subreddit name or URL")

    # Fetch RSS feed
    rss_url = f"https://www.reddit.com/r/{subreddit_clean}/.rss"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise HTTPException(status_code=404, detail="Subreddit not found or RSS feed empty")

    entries = feed.entries[:req.top_n]
    titles = [entry.title for entry in entries]

    # Sentiment Analysis
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for sentiment in (sentiment_analyzer(title)[0]['label'] for title in titles):
        counts[sentiment] += 1

    # Generate sentiment chart
    chart_base64 = generate_sentiment_chart(counts, subreddit_clean, req.top_n)

    # Generate Mistral summary
    posts_text = " ".join(titles)
    prompt = f"IN exactly 3 short sentences summarise the overall sentiment of the following reddit posts :\n{posts_text}."
    summary = query_mistral(prompt, max_tokens=req.max_tokens, temperature=req.temperature)

    return JSONResponse(content={
        "summary": summary,
        "sentiment_graph": chart_base64,
    })
