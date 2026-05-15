import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def landing():
    status_color = "#10B981" if model_loaded else "#EA580C"
    status_text = "Ready" if model_loaded else "Loading models…"
    status_dot = "#10B981" if model_loaded else "#EA580C"
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amazon Reviews — Sentiment + Clustering API</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #F6F9FC;
            color: #334155;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }}
        .card {{
            background: #fff;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 40px;
            max-width: 560px;
            width: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 28px;
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: {status_dot};
            flex-shrink: 0;
        }}
        .title {{
            font-size: 20px;
            font-weight: 700;
            color: #1E293B;
            line-height: 1.3;
        }}
        .subtitle {{
            font-size: 14px;
            color: #64748B;
            font-weight: 400;
            margin-top: 2px;
        }}
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #F1F5F9;
            border-radius: 9999px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 500;
            color: {status_color};
            margin-bottom: 24px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-label {{
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .endpoint {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            margin-bottom: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }}
        .method {{
            font-weight: 700;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .get {{ background: #ECFDF5; color: #10B981; }}
        .post {{ background: #FEF3C7; color: #D97706; }}
        .path {{ color: #334155; }}
        .desc {{ color: #94A3B8; font-size: 13px; margin-left: auto; }}
        textarea {{
            width: 100%;
            min-height: 80px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #334155;
            resize: vertical;
        }}
        textarea:focus {{
            outline: none;
            border-color: #0891B2;
            box-shadow: 0 0 0 3px rgba(8,145,178,0.1);
        }}
        button {{
            background: #0891B2;
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }}
        button:hover {{ background: #0E7490; }}
        button:disabled {{ background: #94A3B8; cursor: not-allowed; }}
        .result {{
            margin-top: 12px;
            padding: 14px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #334155;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            display: none;
        }}
        .result.error {{ color: #DC2626; }}
        .footer {{
            margin-top: 28px;
            padding-top: 20px;
            border-top: 1px solid #E2E8F0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .footer a {{
            color: #0891B2;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
        }}
        .footer a:hover {{ text-decoration: underline; }}
        .models {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .model-tag {{
            background: #F1F5F9;
            color: #475569;
            font-size: 12px;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div class="dot"></div>
            <div>
                <div class="title">Amazon Reviews — Sentiment + Clustering</div>
                <div class="subtitle">DistilBERT · nomic-embed · K-Means centroids</div>
            </div>
        </div>

        <div class="status-badge">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{status_dot}"></span>
            {status_text}
        </div>

        <div class="section">
            <div class="section-label">Endpoints</div>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="path">/health</span>
                <span class="desc">Model status</span>
            </div>
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/predict</span>
                <span class="desc">Batch inference</span>
            </div>
        </div>

        <div class="section">
            <div class="section-label">Quick test</div>
            <textarea id="testInput" placeholder='["This product is amazing!", "Terrible quality, broke in a day.", "It works fine I guess."]'></textarea>
            <div style="margin-top:10px; display:flex; gap:8px; align-items:center;">
                <button id="testBtn" onclick="testPredict()" {'disabled' if not model_loaded else ''}>Test /predict</button>
                <span id="testStatus" style="font-size:12px;color:#94A3B8;"></span>
            </div>
            <pre id="result" class="result"></pre>
        </div>

        <div class="footer">
            <div class="models">
                <span class="model-tag">DistilBERT 66M</span>
                <span class="model-tag">nomic-embed 137M</span>
                <span class="model-tag">6 clusters</span>
            </div>
            <a href="https://project-wmh9z.vercel.app" target="_blank">Open Dashboard →</a>
        </div>
    </div>

    <script>
        async function testPredict() {{
            const input = document.getElementById('testInput').value.trim();
            const result = document.getElementById('result');
            const btn = document.getElementById('testBtn');
            const status = document.getElementById('testStatus');

            if (!input) return;

            btn.disabled = true;
            status.textContent = '…';
            result.style.display = 'block';
            result.textContent = '';
            result.className = 'result';

            try {{
                const texts = JSON.parse(input);
                if (!Array.isArray(texts)) throw new Error('Input must be a JSON array');

                status.textContent = 'Sending…';
                const t0 = performance.now();
                const res = await fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ texts }})
                }});
                const t1 = performance.now();

                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.detail || 'Request failed');
                }}

                const data = await res.json();
                status.textContent = (t1 - t0).toFixed(0) + ' ms';
                result.textContent = JSON.stringify(data, null, 2);
            }} catch (e) {{
                status.textContent = 'Error';
                result.className = 'result error';
                result.textContent = e.message;
            }} finally {{
                btn.disabled = false;
            }}
        }}
    </script>
</body>
</html>"""


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
