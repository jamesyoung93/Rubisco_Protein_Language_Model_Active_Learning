#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch WT Rubisco large-subunit (RbcL / CbbL) sequences for strains listed in a doubling-time CSV.

Inputs
------
--compiled_csv : CSV with a 'strain' column (e.g., cyano_doubling_times_compiled_best_plus_pcc11801.csv)

Outputs
-------
--out_fasta : multi-FASTA of sequences
--out_meta  : CSV with strain, chosen accession, length, protein name, organism name, query used
--out_missing : text file listing strains that could not be resolved

Notes
-----
- Uses UniProt REST endpoints. If your cluster blocks outbound HTTPS, run locally or via a proxy.
- Heuristics:
  - prefer reviewed entries if available
  - require protein name to contain "ribulose" and "carboxylase" (case-insensitive)
  - require length >= 400 aa (filters fragments)
"""

import argparse
import csv
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def uniprot_search(query: str, timeout: int = 60) -> List[Dict]:
    """
    Return a list of UniProt records (JSON) for the query.
    """
    params = {
        "query": query,
        "format": "json",
        "size": 25,
        # request just enough fields to pick a good hit
        "fields": "accession,reviewed,protein_name,organism_name,length,gene_names",
    }
    r = requests.get(UNIPROT_SEARCH, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def pick_best_hit(results: List[Dict]) -> Optional[Dict]:
    """
    Heuristic selection of the best RbcL candidate among UniProt hits.
    """
    def get_field(d, path, default=None):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    candidates = []
    for rec in results:
        acc = rec.get("primaryAccession")
        reviewed = bool(rec.get("entryType") == "UniProtKB reviewed (Swiss-Prot)")
        length = rec.get("sequence", {}).get("length", None)

        prot_name = get_field(rec, ["proteinDescription", "recommendedName", "fullName", "value"], "") or ""
        if not prot_name:
            # sometimes only submittedName exists
            sub = get_field(rec, ["proteinDescription", "submissionNames"], [])
            if sub and isinstance(sub, list):
                prot_name = sub[0].get("fullName", {}).get("value", "") or ""

        org_name = rec.get("organism", {}).get("scientificName", "") or ""

        gene_names = []
        g = rec.get("genes", [])
        if isinstance(g, list):
            for gi in g:
                gn = gi.get("geneName", {}).get("value")
                if gn:
                    gene_names.append(gn)

        # Filters
        if length is None or length < 400:
            continue
        pname_l = prot_name.lower()
        if ("ribulose" not in pname_l) or ("carboxylase" not in pname_l):
            continue

        # Score
        score = 0.0
        if reviewed:
            score += 10.0
        # gene bonus
        if any(x.lower() in ("rbcl", "cbbl") for x in gene_names):
            score += 5.0
        # longer is slightly preferred (avoids truncations)
        score += float(length) / 1000.0

        candidates.append((score, acc, length, prot_name, org_name, gene_names, reviewed))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    return {
        "accession": best[1],
        "length": best[2],
        "protein_name": best[3],
        "organism_name": best[4],
        "gene_names": ";".join(best[5]),
        "reviewed": bool(best[6]),
        "score": best[0],
    }


def fetch_fasta(acc: str, timeout: int = 60) -> str:
    url = UNIPROT_FASTA.format(acc=acc)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text.strip() + "\n"


def normalize_strain_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_query(strain: str) -> str:
    """
    UniProt query: organism exact-ish + (rbcL/cbbL) + protein name keyword.
    """
    # Keep it permissive: strain strings can be messy
    # Try organism:"<strain>" AND (gene:rbcl OR gene:cbbl OR protein:"ribulose bisphosphate carboxylase large chain")
    return (
        f'(organism_name:"{strain}" OR organism:"{strain}") AND '
        f'(gene:rbcl OR gene:cbbl OR protein:"ribulose bisphosphate carboxylase large chain")'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled_csv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_missing", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.compiled_csv)
    if "strain" not in df.columns:
        raise ValueError(f"Expected 'strain' column in {args.compiled_csv}, found: {list(df.columns)}")

    strains = sorted({normalize_strain_name(x) for x in df["strain"].dropna().tolist() if str(x).strip()})
    if not strains:
        raise ValueError("No strains found.")

    meta_rows = []
    missing = []

    with open(args.out_fasta, "w", encoding="utf-8") as f_fa:
        for strain in strains:
            q = build_query(strain)
            try:
                hits = uniprot_search(q)
                best = pick_best_hit(hits)
                if best is None:
                    missing.append(strain)
                    continue

                fasta = fetch_fasta(best["accession"])
                # Rewrite FASTA header to carry strain label
                fasta_lines = fasta.splitlines()
                if fasta_lines and fasta_lines[0].startswith(">"):
                    fasta_lines[0] = f">{strain} | UniProt:{best['accession']} | {best['protein_name']} | {best['organism_name']}"
                f_fa.write("\n".join(fasta_lines) + "\n")

                meta_rows.append({
                    "strain": strain,
                    "accession": best["accession"],
                    "length": best["length"],
                    "reviewed": best["reviewed"],
                    "gene_names": best["gene_names"],
                    "protein_name": best["protein_name"],
                    "organism_name": best["organism_name"],
                    "query": q,
                    "score": best["score"],
                })

            except Exception as e:
                missing.append(strain)
                print(f"[warn] failed strain={strain}: {e}", file=sys.stderr)

    pd.DataFrame(meta_rows).to_csv(args.out_meta, index=False)

    with open(args.out_missing, "w", encoding="utf-8") as f:
        for s in missing:
            f.write(s + "\n")

    print(f"[done] wrote FASTA: {args.out_fasta}")
    print(f"[done] wrote metadata: {args.out_meta}")
    print(f"[done] missing strains: {len(missing)} (see {args.out_missing})")


if __name__ == "__main__":
    main()
