# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Fetch WT Rubisco large-subunit (RbcL / CbbL) protein sequences from UniProt
for strains listed in the compiled cyanobacterial doubling-time CSV.

Outputs:
  - FASTA of selected sequences
  - metadata CSV describing the chosen UniProt entry per strain
  - missing strains list

Notes:
  - Uses robust query fallbacks to avoid UniProt 400 errors.
  - Picks the "best" hit by a simple scoring heuristic (reviewed, length, gene name, protein name).
"""

import argparse
import io
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# Hand-tuned aliases so your compiled "PCC6803" style rows can still be searched sanely.
# You can extend this as you add more strains.
ALIASES: Dict[str, List[str]] = {
    "PCC6803": ["Synechocystis sp. PCC 6803", "Synechocystis sp. (strain PCC 6803)"],
    "Synechocystis sp. PCC 6803": ["Synechocystis sp. PCC 6803", "Synechocystis sp. (strain PCC 6803)"],

    "PCC7002": ["Synechococcus sp. PCC 7002", "Synechococcus sp. (strain PCC 7002)"],
    "Synechococcus sp.  PCC 7002": ["Synechococcus sp. PCC 7002", "Synechococcus sp. (strain PCC 7002)"],
    "Synechococcus sp. PCC 7002": ["Synechococcus sp. PCC 7002", "Synechococcus sp. (strain PCC 7002)"],

    "PCC7942": ["Synechococcus elongatus PCC 7942", "Synechococcus elongatus (strain PCC 7942)"],
    "Synechococcus elongatus PCC 7942": ["Synechococcus elongatus PCC 7942", "Synechococcus elongatus (strain PCC 7942)"],

    "PCC6301": ["Synechococcus elongatus PCC 6301", "Synechococcus elongatus (strain PCC 6301)"],

    "PCC11801": ["Synechococcus elongatus PCC 11801", "Synechococcus elongatus (strain PCC 11801)"],

    "UTEX2973": ["Synechococcus elongatus UTEX 2973", "Synechococcus elongatus (strain UTEX 2973)"],
    "Synechococcus elongatus UTEX 2973": ["Synechococcus elongatus UTEX 2973", "Synechococcus elongatus (strain UTEX 2973)"],

    "Anabaena sp. PCC 7120": ["Anabaena sp. PCC 7120", "Nostoc sp. PCC 7120"],
    "Anabaena sp. PCC 7120": ["Anabaena sp. PCC 7120", "Nostoc sp. PCC 7120"],
    "Anabaena sp. PCC 7120": ["Anabaena sp. PCC 7120", "Nostoc sp. PCC 7120"],
}


def canonicalize_strain(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def uniprot_search_tsv(query: str, size: int = 25, timeout_s: int = 60) -> pd.DataFrame:
    # TSV is easier + more stable than JSON for simple ranking.
    params = {
        "query": query,
        "format": "tsv",
        "size": size,
        "fields": "accession,reviewed,protein_name,organism_name,length,gene_names",
    }
    r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    # Normalize column names across UniProt variants
    colmap = {
        "Entry": "accession",
        "Reviewed": "reviewed",
        "Protein names": "protein_name",
        "Organism": "organism_name",
        "Length": "length",
        "Gene Names": "gene_names",
    }
    df = df.rename(columns={c: colmap.get(c, c) for c in df.columns})
    return df


def score_hit(row: pd.Series) -> float:
    score = 0.0

    reviewed = str(row.get("reviewed", "")).strip().lower()
    if reviewed in {"reviewed", "true", "yes", "1"}:
        score += 10.0

    try:
        L = int(row.get("length", -1))
    except Exception:
        L = -1

    # Form I RbcL is typically ~475 aa (give it a wide window).
    if 420 <= L <= 550:
        score += 5.0
    elif 350 <= L <= 650:
        score += 1.0

    genes = str(row.get("gene_names", "")).lower()
    if "rbcl" in genes:
        score += 3.0
    if "cbbl" in genes:
        score += 3.0
    if "cbbm" in genes:
        score += 1.0  # Form II large subunit, fallback

    pname = str(row.get("protein_name", "")).lower()
    if "ribulose" in pname and ("carboxylase" in pname or "carboxylation" in pname):
        score += 2.0
    if "large subunit" in pname or "large chain" in pname:
        score += 1.0

    return score


def fetch_fasta(accession: str, timeout_s: int = 60) -> str:
    url = UNIPROT_FASTA_URL.format(acc=accession)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def build_queries(strain: str) -> List[str]:
    strain = canonicalize_strain(strain)
    names = ALIASES.get(strain, [strain])

    queries: List[str] = []
    for n in names:
        # Most specific: organism_name + gene constraint
        queries.append(f'organism_name:"{n}" AND (gene:rbcl OR gene:cbbl OR gene:cbbm)')
        # Slightly broader: free-text organism string + gene constraint
        queries.append(f'"{n}" AND (gene:rbcl OR gene:cbbl OR gene:cbbm)')
        # Broadest: just the token(s) + gene constraint
        queries.append(f'{n} AND (gene:rbcl OR gene:cbbl OR gene:cbbm)')

    # De-duplicate while preserving order
    out = []
    seen = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled_csv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_missing", required=True)
    ap.add_argument("--sleep_s", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_fasta) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_meta) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_missing) or ".", exist_ok=True)

    df = pd.read_csv(args.compiled_csv)
    if "strain" not in df.columns:
        raise ValueError("Expected a 'strain' column in compiled CSV.")

    strains = sorted({canonicalize_strain(s) for s in df["strain"].dropna().tolist()})

    metas = []
    missing = []

    with open(args.out_fasta, "w", encoding="utf-8") as f_out:
        for strain in strains:
            best_row = None
            best_score = -1e9
            best_query = None

            for q in build_queries(strain):
                try:
                    hits = uniprot_search_tsv(q, size=25)
                except Exception as e:
                    # Query could still fail in edge cases; continue to next fallback.
                    continue

                if hits is None or len(hits) == 0:
                    continue

                for _, row in hits.iterrows():
                    sc = score_hit(row)
                    if sc > best_score:
                        best_score = sc
                        best_row = row
                        best_query = q

                if best_row is not None and best_score >= 10.0:
                    # "Good enough" early exit: reviewed + plausible length.
                    break

            if best_row is None:
                missing.append(strain)
                continue

            acc = str(best_row.get("accession", "")).strip()
            if not acc:
                missing.append(strain)
                continue

            try:
                fasta = fetch_fasta(acc)
            except Exception:
                missing.append(strain)
                continue

            # Rewrite FASTA header to use the strain as the ID (for downstream merging),
            # but keep UniProt details in the description.
            fasta_lines = fasta.strip().splitlines()
            seq = "".join([ln.strip() for ln in fasta_lines[1:] if ln.strip()])
            desc = f"uniprot={acc} organism={best_row.get('organism_name','')} length={best_row.get('length','')} genes={best_row.get('gene_names','')}"
            f_out.write(f">{strain} {desc}\n")
            # Wrap at 80 chars
            for i in range(0, len(seq), 80):
                f_out.write(seq[i:i+80] + "\n")

            metas.append({
                "strain": strain,
                "uniprot_accession": acc,
                "query_used": best_query,
                "score": float(best_score),
                "reviewed": best_row.get("reviewed", ""),
                "protein_name": best_row.get("protein_name", ""),
                "organism_name": best_row.get("organism_name", ""),
                "length": best_row.get("length", ""),
                "gene_names": best_row.get("gene_names", ""),
            })

            time.sleep(max(0.0, args.sleep_s))

    pd.DataFrame(metas).to_csv(args.out_meta, index=False)
    with open(args.out_missing, "w", encoding="utf-8") as f_m:
        for s in missing:
            f_m.write(s + "\n")

    print(f"[done] wrote FASTA: {args.out_fasta}")
    print(f"[done] wrote metadata: {args.out_meta}")
    print(f"[done] missing strains: {len(missing)} (see {args.out_missing})")


if __name__ == "__main__":
    main()
