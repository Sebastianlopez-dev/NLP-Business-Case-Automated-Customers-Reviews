# ============================================================
# N02.1 — Export DistilBERT Sentiment to ONNX (Transformers.js)
# Usage: Copy each cell into Google Colab and run in order
# Output: data/models/distilbert-sentiments/onnx/
# ============================================================

# ============================================================
# CELL 0: Environment Detection & Drive Mount
# ============================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Colab: {IN_COLAB}")

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Drive mounted.")
else:
    print("⚠️  Not in Colab — adjust BASE_DIR below.")

# ============================================================
# CELL 1: Paths (cell-paths)
# ============================================================
BASE_DIR    = "/content/drive/MyDrive/nlp-project/business-case-01"
MODEL_DIR   = f"{BASE_DIR}/data/models/distilbert-sentiments"
ONNX_DIR    = f"{MODEL_DIR}/onnx"

import os
print(f"BASE_DIR  : {BASE_DIR}")
print(f"MODEL_DIR : {MODEL_DIR}")
print(f"ONNX_DIR  : {ONNX_DIR}")
print(f"Model exists : {os.path.isdir(MODEL_DIR)}")
print(f"Files: {os.listdir(MODEL_DIR)[:5]}")

# ============================================================
# CELL 2: Install Dependencies (~1 min)
# ============================================================
!pip install -q optimum[exporters] onnx onnxruntime

# ============================================================
# CELL 3: Export Model to ONNX (~2-3 min)
# ============================================================
!optimum-cli export onnx \
  --model "{MODEL_DIR}" \
  --task text-classification \
  "{ONNX_DIR}"

# ============================================================
# CELL 4: Verify Output
# ============================================================
import os
files = os.listdir(ONNX_DIR)
sizes = {f: os.path.getsize(f"{ONNX_DIR}/{f}") / 1e6 for f in files}
print(f"✅ ONNX exported to: {ONNX_DIR}")
for f, s in sizes.items():
    print(f"   {f:<30s} {s:.1f} MB")
print(f"\n⬆️  Upload the onnx/ folder to HF Hub:")
print(f"   Repo: SebasLopez-ai/distilbert-amazon-reviews-sentiment")
print(f"   Path: onnx/")
