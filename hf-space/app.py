import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time

# ── Global state ──
model_loaded = False
sentiment_pipe = None
embedder = None
centroids = None
LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


def load_models():
    """Load all models (blocking — called in background thread)."""
    global model_loaded, sentiment_pipe, embedder, centroids

    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    centroids = np.load("cluster_centroids.npy")
    assert centroids.shape == (6, 768), f"Expected (6, 768), got {centroids.shape}"

    sentiment_pipe = pipeline(
        "text-classification",
        model="SebasLopez-ai/distilbert-amazon-reviews-sentiment",
    )
    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
    model_loaded = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models in a thread so uvicorn can start serving /health immediately
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_models)
    yield


app = FastAPI(title="Amazon Reviews Sentiment + Clustering", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


@app.get("/health")
def health():
    return {"model_loaded": model_loaded}


@app.post("/predict")
def predict(req: PredictRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    from sklearn.metrics.pairwise import cosine_similarity

    start = time.time()

    sentiments = sentiment_pipe(req.texts)
    embeddings = embedder.encode(req.texts, convert_to_numpy=True)
    clusters = np.argmax(cosine_similarity(embeddings, centroids), axis=1)

    predictions = [
        {
            "label": LABEL_MAP.get(s["label"], 1),
            "score": float(s["score"]),
            "cluster": int(c),
        }
        for s, c in zip(sentiments, clusters)
    ]

    return {
        "predictions": predictions,
        "model_loaded": True,
        "inference_time_ms": int((time.time() - start) * 1000),
    }
