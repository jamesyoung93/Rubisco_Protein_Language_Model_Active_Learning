#!/usr/bin/env python3
import argparse, pandas as pd, re

def canon(s): return re.sub(r"\s+"," ",str(s).strip()).lower()

ALIASES = {
  "pcc7942":"synechococcus elongatus pcc 7942",
  "pcc6803":"synechocystis sp. pcc 6803",
  "pcc7002":"synechococcus sp. pcc 7002",
  "pcc6301":"synechococcus elongatus pcc 6301",
  "utex2973":"synechococcus elongatus utex 2973",
  "pcc11801":"synechococcus elongatus pcc 11801",
}

ap = argparse.ArgumentParser()
ap.add_argument("--flamholz_csv", required=True)
ap.add_argument("--strain_csv", required=True)
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()

fl = pd.read_csv(args.flamholz_csv)
sc = pd.read_csv(args.strain_csv)

# Guess organism column
org_col = None
for c in fl.columns:
    if "organism" in c.lower() or "species" in c.lower() or "taxon" in c.lower():
        org_col = c
        break
if org_col is None:
    raise SystemExit(f"Could not find organism/species column. Columns: {list(fl.columns)}")

fl["_org"] = fl[org_col].map(canon)

rows=[]
for s in sorted(set(sc["strain"].dropna().astype(str))):
    q = canon(s)
    q = canon(ALIASES.get(q, q))
    hits = fl[fl["_org"].str.contains(q, na=False)]
    rows.append({"strain": s, "query": q, "n_hits": len(hits)})
out = pd.DataFrame(rows).sort_values("n_hits", ascending=False)
out.to_csv(args.out_csv, index=False)
print(out.to_string(index=False))
