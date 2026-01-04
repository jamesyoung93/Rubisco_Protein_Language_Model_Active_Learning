#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
import pandas as pd
import numpy as np

def canon(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.lower()

# Map your strain labels -> Flamholz S1 "species" strings (canonicalized)
MAP = {
    "pcc7942": "synechococcus elongatus pcc 7942",
    "synechococcus elongatus pcc 7942": "synechococcus elongatus pcc 7942",

    "pcc7002": "synechococcus 7002",
    "synechococcus sp. pcc 7002": "synechococcus 7002",
    "synechococcus sp.  pcc 7002": "synechococcus 7002",

    "pcc6301": "synechococcus 6301",
    "synechococcus elongatus pcc 6301": "synechococcus 6301",

    "anabaena sp. pcc 7120": "anabaena pcc7120",
    "anabaena pcc7120": "anabaena pcc7120",
}

KIN_COLS = ["KC", "vC", "S", "KO", "vO", "KRuBP", "temp_C", "pH"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flamholz_csv", required=True)
    ap.add_argument("--strain_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--wt_only", action="store_true", help="Filter to WT-ish rows: primary=1, mutant=0, heterologous_expression==False.")
    args = ap.parse_args()

    fl = pd.read_csv(args.flamholz_csv)
    sc = pd.read_csv(args.strain_csv)

    if "species" not in fl.columns:
        raise SystemExit(f"Expected a 'species' column in Flamholz CSV. Columns: {list(fl.columns)}")

    fl["_species_c"] = fl["species"].map(canon)

    if args.wt_only:
        # these columns exist in your S1 file; keep rows that look like native WT measurements
        for col in ["primary", "mutant", "heterologous_expression"]:
            if col not in fl.columns:
                raise SystemExit(f"--wt_only requested but '{col}' not found in Flamholz CSV.")
        fl = fl[(fl["primary"] == 1) & (fl["mutant"] == 0) & (fl["heterologous_expression"] == False)].copy()

    # build a list of target species strings from your strain table
    strains = sorted(set(sc["strain"].dropna().astype(str)))
    rows = []

    for s in strains:
        key = canon(s)
        mapped = MAP.get(key, None)

        # if not in MAP, fall back to searching just digits like 7002/6301/7942/7120
        if mapped is None:
            m = re.search(r"(7002|6301|7942|7120)", key)
            if m:
                mapped = canon(m.group(1))

        if mapped is None:
            rows.append({"strain": s, "mapped_species": "", "n_rows": 0})
            continue

        # Exact match against canonical species strings
        hits = fl[fl["_species_c"] == mapped].copy()

        # Secondary fallback: contains, to catch e.g., "Synechococcus 6301 ERD1 mutant" if wt_only is off
        if hits.empty and not args.wt_only:
            hits = fl[fl["_species_c"].str.contains(mapped, na=False)].copy()

        if hits.empty:
            rows.append({"strain": s, "mapped_species": mapped, "n_rows": 0})
            continue

        # summarize with medians (robust to multiple measurements)
        summary = {"strain": s, "mapped_species": mapped, "n_rows": len(hits)}
        for c in KIN_COLS:
            if c in hits.columns:
                summary[c] = float(np.nanmedian(pd.to_numeric(hits[c], errors="coerce").to_numpy()))
        rows.append(summary)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out.sort_values("n_rows", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
