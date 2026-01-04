# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Embed protein sequences in a FASTA file using ESM2, producing a fixed-length
vector per sequence via mean pooling.

Outputs a .npy file containing a dict:
  {"ids": np.array([...]), "emb": np.ndarray[n, d]}

This format matches the "ids/emb" pattern used throughout your Rubisco pipeline.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    raise ImportError("This script requires torch. Install torch or load a module providing it.") from e

try:
    import esm
except Exception as e:
    raise ImportError(
        "This script requires fair-esm. Try: pip install fair-esm"
    ) from e


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    cur_id = None
    cur_seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq)))
                header = line[1:].strip()
                # ID is first token; in our fetch script it is the strain string up to the first space.
                cur_id = header.split("  ")[0].split("\t")[0].split(" ", 1)[0] if header else "unknown"
                # But we actually want the full strain ID including spaces in the strain name.
                # Because our fetch script writes: >{strain} uniprot=...
                # We can recover strain as the substring before " uniprot=":
                if " uniprot=" in header:
                    cur_id = header.split(" uniprot=")[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            records.append((cur_id, "".join(cur_seq)))
    return records


def load_model(model_name: str):
    loaders = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
    }
    if model_name not in loaders:
        raise ValueError(f"Unsupported model_name={model_name}. Choose one of: {list(loaders)}")
    model, alphabet = loaders[model_name]()
    return model, alphabet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--repr_layer", type=int, default=33)
    ap.add_argument("--pool", choices=["mean", "cls"], default="mean")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    records = read_fasta(args.fasta)
    if len(records) == 0:
        raise ValueError(f"No FASTA records found in {args.fasta}")

    model, alphabet = load_model(args.model)
    batch_converter = alphabet.get_batch_converter()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model = model.to(device)
    model.eval()

    ids = []
    embs = []

    with torch.no_grad():
        for seq_id, seq in records:
            # Defensive cleanup: keep only AA letters and remove gaps/spaces
            seq = seq.replace(" ", "").replace("-", "")
            ids.append(seq_id)

            data = [(seq_id, seq)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)

            out = model(toks, repr_layers=[args.repr_layer], return_contacts=False)
            token_reps = out["representations"][args.repr_layer][0]  # [L, d]

            # Determine sequence length excluding padding, BOS, EOS
            # tokens: [BOS] ... [EOS] [PAD]...
            pad = alphabet.padding_idx
            nonpad = (toks[0] != pad).sum().item()
            # nonpad includes BOS and EOS
            seqlen = max(0, nonpad - 2)

            if args.pool == "cls":
                emb = token_reps[0].detach().cpu().numpy()
            else:
                # mean over residue positions only
                emb = token_reps[1:1+seqlen].mean(dim=0).detach().cpu().numpy()

            embs.append(emb.astype(np.float32))

    emb_mat = np.vstack(embs).astype(np.float32)
    out = {"ids": np.array(ids, dtype=object), "emb": emb_mat}
    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, out)

    print(f"[done] wrote embeddings: {args.out_npy}  shape={emb_mat.shape}")


if __name__ == "__main__":
    main()
