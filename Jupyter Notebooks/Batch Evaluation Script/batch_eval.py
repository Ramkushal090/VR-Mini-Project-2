"""
Batch Evaluation Script — Deliverable #3
=========================================
Given a folder of query images, runs the retrieval pipeline end-to-end
and computes: Recall@K, NDCG@K, mAP@K  for K ∈ {5, 10, 15}.

TWO MODES:
  Mode 1 — DeepFashion partition (default):
      python batch_eval.py
      Uses list_eval_partition.txt to find query images + derives item_id
      from folder path. No extra files needed.

  Mode 2 — External ground truth CSV:
      python batch_eval.py --gt_csv /path/to/gt.csv --query_dir /path/to/images/
      CSV must have columns: query_image, item_id
        query_image  → filename (e.g. jacket_test1.jpg) or relative path
        item_id      → ground truth item id (e.g. id_00001234)

Ground truth rule: two images match iff they share the same item_id.
"""

import os, json, torch, faiss, numpy as np, pandas as pd, argparse
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 0. Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch Evaluation Script")
parser.add_argument(
    "--gt_csv",
    type=str,
    default=None,
    help="(Mode 2) Path to ground truth CSV with columns: query_image, item_id"
)
parser.add_argument(
    "--query_dir",
    type=str,
    default=None,
    help="(Mode 2) Folder containing the query images referenced in --gt_csv"
)
args = parser.parse_args()

MODE = 2 if args.gt_csv else 1
print(f"\n{'='*55}")
print(f"  Batch Evaluation — Mode {MODE}")
if MODE == 1:
    print("  Using DeepFashion partition file for ground truth")
else:
    print(f"  Using external GT CSV : {args.gt_csv}")
    print(f"  Query image folder    : {args.query_dir}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Paths  (adjust to your Kaggle layout)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR      = '/kaggle/input/datasets/hades199/vr-final-project-dataset/vr-final-project-dataset'
PARTITION_FILE   = os.path.join(DATASET_DIR, 'Eval/list_eval_partition.txt')
BBOX_FILE        = os.path.join(DATASET_DIR, 'Anno/list_bbox_inshop.txt')
CROPPED_DIR      = '/kaggle/input/datasets/hades199/3c-yolo-cropped-images'
CLIP_MODEL_PATH  = '/kaggle/input/datasets/hades199/3c-clip-fintuned-model/clip_finetuned_v2/seed_623/best'
FAISS_INDEX_PATH = '/kaggle/input/datasets/hades199/3c-faiss-indexes/faiss_indexes/C_finetuned_seed623_alpha0.7_index.bin'
FAISS_META_PATH  = '/kaggle/input/datasets/hades199/3c-faiss-indexes/faiss_indexes/C_finetuned_seed623_alpha0.7_metadata.json'

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [5, 10, 15]
MAX_K    = max(K_VALUES)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build query list  (different for each mode)
# ─────────────────────────────────────────────────────────────────────────────
def extract_item_id(name):
    for part in name.replace('\\', '/').split('/'):
        if part.startswith('id_'):
            return part
    return None

if MODE == 1:
    # ── Mode 1: DeepFashion partition ────────────────────────────────────────
    print("Loading DeepFashion partition data...")
    df_part = pd.read_csv(PARTITION_FILE, sep=r'\s+', skiprows=1)
    df_bbox = pd.read_csv(BBOX_FILE,      sep=r'\s+', skiprows=1)
    df_part = pd.merge(df_part, df_bbox[['image_name', 'clothes_type']], on='image_name', how='left')
    df_part['item_id'] = df_part['image_name'].apply(extract_item_id)
    df_part = df_part.dropna(subset=['item_id'])

    df_query = df_part[df_part['evaluation_status'] == 'query'].reset_index(drop=True)
    print(f"  Query images : {len(df_query)}")

    def get_image_path(row):
        """Try GT crop first, then raw DeepFashion image."""
        image_name = row['image_name']
        p1 = os.path.join(CROPPED_DIR, image_name)
        p2 = os.path.join(CROPPED_DIR, image_name.replace('/', '_'))
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
        raw = os.path.join(DATASET_DIR, image_name)
        if os.path.exists(raw): return raw
        return None

    def get_query_item_id(row):
        return row['item_id']

else:
    # ── Mode 2: External GT CSV ───────────────────────────────────────────────
    if not args.query_dir:
        raise ValueError("--query_dir is required when using --gt_csv (Mode 2)")
    if not os.path.exists(args.gt_csv):
        raise FileNotFoundError(f"GT CSV not found: {args.gt_csv}")

    print("Loading external ground truth CSV...")
    df_query = pd.read_csv(args.gt_csv)

    # Validate columns
    required = {'query_image', 'item_id'}
    if not required.issubset(df_query.columns):
        raise ValueError(
            f"GT CSV must have columns: {required}. "
            f"Found: {set(df_query.columns)}"
        )

    df_query = df_query.dropna(subset=['query_image', 'item_id']).reset_index(drop=True)
    print(f"  Query images in CSV: {len(df_query)}")

    def get_image_path(row):
        """Resolve image path from query_dir."""
        fname = row['query_image']
        # Try as-is, then as basename
        p1 = os.path.join(args.query_dir, fname)
        p2 = os.path.join(args.query_dir, os.path.basename(fname))
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
        # Direct absolute path
        if os.path.exists(fname): return fname
        return None

    def get_query_item_id(row):
        return row['item_id']

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load CLIP + FAISS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
clip_model.eval()

print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
try:
    hnsw = faiss.downcast_index(faiss_index.index)
    hnsw.hnsw.efSearch = 128
except Exception:
    pass  # Index may not be an IndexIDMap wrapping HNSW; search still works

with open(FAISS_META_PATH, 'r') as f:
    metadata = json.load(f)

# Build reverse map: faiss_id → item_id
faiss_id_to_item = {int(fid): meta.get('item_id', '') for fid, meta in metadata.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper: CLIP embedding
# ─────────────────────────────────────────────────────────────────────────────
def get_clip_embedding(image):
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        v_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        emb   = clip_model.visual_projection(v_out.pooler_output)
    return F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Metric functions
# ─────────────────────────────────────────────────────────────────────────────
def recall_at_k(relevant, retrieved_ids, k):
    """1 if any retrieved item in top-k matches ground truth, else 0."""
    return 1.0 if any(rid in relevant for rid in retrieved_ids[:k]) else 0.0

def average_precision_at_k(relevant, retrieved_ids, k):
    """AP@K for one query."""
    hits, score = 0, 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0

def ndcg_at_k(relevant, retrieved_ids, k):
    """NDCG@K for one query (binary relevance)."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
        if rid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning batch evaluation on {len(df_query)} queries...")
print(f"K values: {K_VALUES}\n")

results = {k: {'recall': [], 'ndcg': [], 'ap': []} for k in K_VALUES}
skipped = 0

for _, row in tqdm(df_query.iterrows(), total=len(df_query), desc="Evaluating"):
    query_item_id = get_query_item_id(row)
    relevant      = {query_item_id}

    # Resolve image path
    img_path = get_image_path(row)
    if not img_path:
        skipped += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        skipped += 1
        continue

    # CLIP embed → FAISS search
    emb     = get_clip_embedding(img)
    D, I    = faiss_index.search(emb, k=MAX_K)
    retrieved_ids = [faiss_id_to_item.get(int(fid), '') for fid in I[0]]

    # Compute metrics at each K
    for k in K_VALUES:
        results[k]['recall'].append(recall_at_k(relevant, retrieved_ids, k))
        results[k]['ndcg'].append(ndcg_at_k(relevant, retrieved_ids, k))
        results[k]['ap'].append(average_precision_at_k(relevant, retrieved_ids, k))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Report
# ─────────────────────────────────────────────────────────────────────────────
evaluated = len(results[K_VALUES[0]]['recall'])
print(f"\nSkipped  : {skipped} queries (image not found)")
print(f"Evaluated: {evaluated} queries")
print(f"\n{'='*60}")
print(f"  Retrieval Metrics — Best Model (Ablation C, seed 623, α=0.7)")
print(f"  Mode {'1 — DeepFashion partition' if MODE == 1 else '2 — External GT CSV'}")
print(f"{'='*60}")
print(f"  {'K':>4}  |  {'Recall@K':>9}  |  {'NDCG@K':>8}  |  {'mAP@K':>8}")
print(f"  {'-'*4}--+-{'-'*9}--+-{'-'*8}--+-{'-'*8}")

rows = []
for k in K_VALUES:
    r = np.mean(results[k]['recall'])
    n = np.mean(results[k]['ndcg'])
    m = np.mean(results[k]['ap'])
    rows.append({'K': k, 'Recall@K': round(r, 4), 'NDCG@K': round(n, 4), 'mAP@K': round(m, 4)})
    print(f"  {k:>4}  |  {r:>9.4f}  |  {n:>8.4f}  |  {m:>8.4f}")

print(f"{'='*60}")

# Save CSV
df_out  = pd.DataFrame(rows)
out_csv = '/kaggle/working/batch_eval_results.csv'
df_out.to_csv(out_csv, index=False)
print(f"\n✅ Results saved to {out_csv}")
