# embed_esm2.py
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden, attention_mask):
    # last_hidden: [B, L, D]; attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--seq_col", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ids = df[args.id_col].astype(str).tolist()
    seqs = df[args.seq_col].astype(str).str.replace(r"\s+", "", regex=True).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i:i+args.batch_size]
            tokens = tok(batch, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens)
            pooled = mean_pool(out.last_hidden_state, tokens["attention_mask"])
            embs.append(pooled.float().cpu().numpy())

    X = np.vstack(embs)
    np.save(args.out_npy, {"ids": np.array(ids), "emb": X}, allow_pickle=True)

if __name__ == "__main__":
    main()
