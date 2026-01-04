#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_rubisco_wt_sequences_v2.py

Fetch WT Rubisco large subunit (RbcL/CbbL) sequences for strains listed in a doubling-time CSV.

Key fix vs v1:
- Uses UniProt Taxonomy REST API to resolve strain -> taxonId
- Uses UniProtKB query field organism_id:<taxonId> (avoids invalid query fields that caused HTTP 400)

Outputs:
- FASTA of selected sequences
- metadata CSV
- missing strains list

Requirements: requests, pandas
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

UNIPROT_TAX_SEARCH = "https://rest.uniprot.org/taxonomy/search"
UNIPROT_KB_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# Small alias map to help taxonomy resolution when the CSV uses shorthand
ALIASES = {
    "PCC6803": "Synechocystis sp. PCC 6803",
    "Synechocystis sp. PCC 6803": "Synechocystis sp. PCC 6803",
    "PCC7942": "Synechococcus elongatus PCC 7942",
    "Synechococcus elongatus PCC 7942": "Synechococcus elongatus PCC 7942",
    "PCC7002": "Synechococcus sp. PCC 7002",
    "Synechococcus sp. PCC 7002": "Synechococcus sp. PCC 7002",
    "PCC6301": "Synechococcus elongatus PCC 6301",
    "UTEX2973": "Synechococcus elongatus UTEX 2973",
    "PCC11801": "Synechococcus elongatus PCC 11801",
    # Historical naming
    "Anabaena sp. PCC 7120": "Nostoc sp. PCC 7120",
    "Anabaena ATCC 29413": "Nostoc sp. ATCC 29413",
}


def norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tax_search(name: str, timeout: int = 60) -> List[Dict]:
    params = {"query": name, "format": "json", "size": 10}
    r = requests.get(UNIPROT_TAX_SEARCH, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", [])


def pick_taxon(results: List[Dict], query_name: str) -> Optional[Tuple[int, str]]:
    """
    Pick a taxon result:
    - Prefer exact/substring matches on scientificName containing the PCC/ATCC/UTEX token.
    - Otherwise take the first result.
    """
    if not results:
        return None

    token = None
    m = re.search(r"(PCC\s?\d+|ATCC\s?\d+|UTEX\s?\d+)", query_name, flags=re.I)
    if m:
        token = m.group(1).replace(" ", "").upper()

    def sci_name(rec: Dict) -> str:
        return (rec.get("scientificName") or rec.get("name") or "").strip()

    # Try to find best match
    if token:
        for rec in results:
            sn = sci_name(rec).replace(" ", "").upper()
            if token in sn:
                tid = rec.get("taxonId")
                if tid is not None:
                    return int(tid), sci_name(rec)

    # fallback first
    rec = results[0]
    tid = rec.get("taxonId")
    if tid is None:
        return None
    return int(tid), sci_name(rec)


def kb_search(taxon_id: int, timeout: int = 60) -> List[Dict]:
    """
    Query UniProtKB for RbcL/CbbL within organism_id:<taxon_id>.
    Use gene_exact when possible; keep it permissive.
    """
    q = f"(organism_id:{taxon_id}) AND (gene_exact:rbcl OR gene_exact:cbbl OR gene:rbcl OR gene:cbbl)"
    params = {
        "query": q,
        "format": "json",
        "size": 25,
        "fields": "accession,reviewed,id,protein_name,gene_names,organism_name,organism_id,length",
    }
    r = requests.get(UNIPROT_KB_SEARCH, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", [])


def extract_protein_name(rec: Dict) -> str:
    pdsc = rec.get("proteinDescription", {}) or {}
    rn = (((pdsc.get("recommendedName") or {}).get("fullName") or {}).get("value")) or ""
    if rn:
        return rn
    subs = pdsc.get("submissionNames") or []
    if subs:
        return ((subs[0].get("fullName") or {}).get("value")) or ""
    return ""


def pick_best_entry(results: List[Dict]) -> Optional[Dict]:
    """
    Pick best RbcL-like entry:
    - length >= 400
    - protein name contains ribulose and carboxylase
    - prefer reviewed
    """
    best = None
    best_score = -1.0

    for rec in results:
        acc = rec.get("primaryAccession")
        length = (rec.get("sequence") or {}).get("length")
        if not acc or not length or length < 400:
            continue

        pname = extract_protein_name(rec).lower()
        if "ribulose" not in pname or "carboxylase" not in pname:
            # still allow some common annotations
            if "rbc" not in pname:
                continue

        reviewed = (rec.get("entryType") == "UniProtKB reviewed (Swiss-Prot)")
        genes = rec.get("genes") or []
        gene_vals = []
        for g in genes:
            v = ((g.get("geneName") or {}).get("value"))
            if v:
                gene_vals.append(v.lower())

        score = 0.0
        if reviewed:
            score += 10.0
        if any(g in ("rbcl", "cbbl") for g in gene_vals):
            score += 5.0
        score += float(length) / 1000.0

        if score > best_score:
            best_score = score
            best = rec

    if best is None:
        return None

    return {
        "accession": best.get("primaryAccession"),
        "reviewed": (best.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"),
        "length": (best.get("sequence") or {}).get("length"),
        "protein_name": extract_protein_name(best),
        "organism_name": (best.get("organism") or {}).get("scientificName", ""),
        "organism_id": (best.get("organism") or {}).get("taxonId", ""),
        "gene_names": ";".join(
            [((g.get("geneName") or {}).get("value") or "") for g in (best.get("genes") or []) if (g.get("geneName") or {}).get("value")]
        ),
        "score": best_score,
    }


def fetch_fasta(acc: str, timeout: int = 60) -> str:
    r = requests.get(UNIPROT_FASTA.format(acc=acc), timeout=timeout)
    r.raise_for_status()
    return r.text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled_csv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_missing", required=True)
    ap.add_argument("--sleep_s", type=float, default=0.25, help="Polite delay between requests.")
    args = ap.parse_args()

    df = pd.read_csv(args.compiled_csv)
    if "strain" not in df.columns:
        raise ValueError("compiled_csv must contain a 'strain' column")

    strains = sorted({norm(x) for x in df["strain"].dropna().tolist() if str(x).strip()})
    meta = []
    missing = []

    with open(args.out_fasta, "w", encoding="utf-8") as f_fa:
        for s in strains:
            qname = ALIASES.get(s, s)

            try:
                tax_hits = tax_search(qname)
                picked = pick_taxon(tax_hits, qname)
                if picked is None:
                    missing.append(s)
                    continue

                tax_id, tax_name = picked

                kb_hits = kb_search(tax_id)
                best = pick_best_entry(kb_hits)
                if best is None or not best.get("accession"):
                    missing.append(s)
                    continue

                fasta = fetch_fasta(best["accession"])
                lines = fasta.splitlines()
                if lines and lines[0].startswith(">"):
                    lines[0] = f">{s} | taxon:{tax_id} | UniProt:{best['accession']} | {best['protein_name']} | {tax_name}"
                f_fa.write("\n".join(lines) + "\n")

                meta.append({
                    "strain_input": s,
                    "tax_query_name": qname,
                    "taxon_id": tax_id,
                    "taxon_name": tax_name,
                    **best,
                })

            except Exception as e:
                print(f"[warn] failed strain={s}: {e}", file=sys.stderr)
                missing.append(s)

            time.sleep(args.sleep_s)

    pd.DataFrame(meta).to_csv(args.out_meta, index=False)
    with open(args.out_missing, "w", encoding="utf-8") as f:
        for s in missing:
            f.write(s + "\n")

    print(f"[done] wrote FASTA: {args.out_fasta}")
    print(f"[done] wrote metadata: {args.out_meta}")
    print(f"[done] missing strains: {len(missing)} (see {args.out_missing})")


if __name__ == "__main__":
    main()
