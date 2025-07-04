import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# ─── Configuration ─────────────────────────────────────
MODEL_ID     = "jinaai/jina-embeddings-v2-base-code"
LOCAL_REPO   = Path(os.environ.get("LOCAL_REPO_PATH", r"C:\Users\vdonc\mco")).resolve()
OUTPUT_DIR   = Path("./data/quantized/jina-int8-dynamic").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load FP32 Model & Tokenizer ───────────────────────
tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID)
model_fp32 = AutoModel.from_pretrained(MODEL_ID)
model_fp32.eval()

# ─── Apply Dynamic Quantization ─────────────────────────
quantized_model = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# ─── Save Quantized Model for HuggingFace Loading ──────
# Saves model files in the format: config.json, pytorch_model.bin, etc.
quantized_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
# Save tokenizer files
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved dynamic-quantized model to {OUTPUT_DIR}")
