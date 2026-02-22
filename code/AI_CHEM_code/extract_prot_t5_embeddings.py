#!/usr/bin/env python3
"""Extract mean-pooled ProtT5 embeddings in the same npy format as embed_esm2.py.

Output format:
    np.save(out_npy, {"ids": np.array(ids), "emb": X}, allow_pickle=True)
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def clean_sequence(seq: str) -> str:
    """ProtT5 expects whitespace-delimited amino acids and uncommon residues mapped to X."""
    seq = "".join(str(seq).split()).upper()
    seq = "".join(ch if ch in "ACDEFGHIKLMNPQRSTVWY" else "X" for ch in seq)
    return " ".join(list(seq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--seq_col", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--model", default="Rostlab/prot_t5_xl_uniref50")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--fp16", action="store_true", help="Use half precision on CUDA")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ids = df[args.id_col].astype(str).tolist()
    seqs = [clean_sequence(s) for s in df[args.seq_col].astype(str).tolist()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model)
    model = model.to(device)
    if device == "cuda" and args.fp16:
        model = model.half()
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i : i + args.batch_size]
            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            embs.append(pooled.float().cpu().numpy())

    X = np.vstack(embs)
    np.save(args.out_npy, {"ids": np.array(ids), "emb": X}, allow_pickle=True)
    print(f"Saved ProtT5 embeddings: {args.out_npy} shape={X.shape}")


if __name__ == "__main__":
    main()
