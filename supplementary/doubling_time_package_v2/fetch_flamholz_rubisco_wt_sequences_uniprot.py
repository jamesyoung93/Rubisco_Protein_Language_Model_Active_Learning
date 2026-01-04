#!/usr/bin/env python3
"""
Fetch WT Rubisco large-chain AA sequences for organisms in Flamholz dataset S1.

Input:
  --flamholz_csv: Flamholz dataset CSV (must contain a 'species' column; S1 has it)
Outputs:
  --out_fasta: FASTA with 1 sequence per species (best UniProt hit)
  --out_meta:  metadata CSV (species, label, accession, gene_names, length, md5, etc.)
  --out_missing: list of species we couldn't resolve
  --cache_json: cache of UniProt search + fetch decisions for reproducibility

Uses UniProt REST API:
  - search: https://rest.uniprot.org/uniprotkb/search
  - fetch fasta: https://rest.uniprot.org/uniprotkb/{accession}.fasta

References:
  UniProt programmatic access supports structured retrieval via REST endpoints. (see UniProt NAR 2025)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

# Rubisco large-chain gene aliases across forms:
DEFAULT_GENE_TERMS = ["rbcl", "cbbl", "cbbm"]
# Protein name variants commonly used in UniProt:
DEFAULT_PROTEIN_TERMS = [
    "ribulose bisphosphate carboxylase large chain",
    "ribulose-1,5-bisphosphate carboxylase/oxygenase large subunit",
]

RETURN_FIELDS = "accession,reviewed,protein_name,organism_name,length,gene_names"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# ESM tolerates X; we map non-standard to X.
def sanitize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq.upper())
    cleaned = []
    for c in seq:
        if c in AA20:
            cleaned.append(c)
        else:
            cleaned.append("X")
    return "".join(cleaned)

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unknown"

def candidate_organism_names(species: str) -> List[str]:
    """
    Generate a small list of increasingly relaxed organism-name candidates
    to handle minor naming differences (e.g., 'Synechococcus 7002' vs 'Synechococcus sp. PCC 7002').
    """
    species = " ".join(str(species).split())
    cands = [species]

    # If "Genus 7002" style, try PCC variants.
    m = re.match(r"^([A-Za-z]+)\s+(\d{4,6})$", species)
    if m:
        genus, num = m.group(1), m.group(2)
        cands.append(f"{genus} PCC {num}")
        cands.append(f"{genus} sp. PCC {num}")

    # If at least Genus species, add just the binomial name as fallback.
    toks = re.split(r"\s+", re.sub(r"[()]+", " ", species)).strip().split() if False else species.split()
    if len(toks) >= 2:
        cands.append(" ".join(toks[:2]))

    # De-duplicate while preserving order
    out = []
    for c in cands:
        c = " ".join(c.split())
        if c and c not in out:
            out.append(c)
    return out

def build_uniprot_query(org_name: str) -> str:
    org_name = org_name.replace('"', "")
    gene_clause = " OR ".join([f"gene:{g}" for g in DEFAULT_GENE_TERMS])
    prot_clause = " OR ".join([f'protein:"{p}"' for p in DEFAULT_PROTEIN_TERMS])
    return f'((organism_name:"{org_name}" OR organism:"{org_name}")) AND (({gene_clause}) OR ({prot_clause}))'

def parse_tsv(tsv_text: str) -> pd.DataFrame:
    if not tsv_text.strip():
        return pd.DataFrame()
    df = pd.read_csv(StringIO(tsv_text), sep="\t")
    return df

def score_hit(row: pd.Series) -> float:
    score = 0.0

    reviewed = str(row.get("Reviewed", row.get("reviewed", ""))).strip().lower()
    if reviewed in {"reviewed", "true", "yes", "1"}:
        score += 3.0

    gene_names = str(row.get("Gene Names", row.get("gene_names", ""))).lower()
    if any(g in gene_names for g in DEFAULT_GENE_TERMS):
        score += 3.0

    protein_name = str(row.get("Protein names", row.get("protein_name", ""))).lower()
    if "ribulose" in protein_name and ("carboxylase" in protein_name or "oxygenase" in protein_name):
        score += 1.0
    if "fragment" in protein_name:
        score -= 3.0

    try:
        length = float(row.get("Length", row.get("length", float("nan"))))
    except Exception:
        length = float("nan")
    if pd.notna(length):
        if 300 <= length <= 650:
            score += 1.0
        else:
            score -= 1.0

    return score

def uniprot_search(org_name: str, session: requests.Session, timeout_s: int = 30) -> pd.DataFrame:
    q = build_uniprot_query(org_name)
    params = {
        "query": q,
        "format": "tsv",
        "fields": RETURN_FIELDS,
        "size": 25,
    }
    r = session.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout_s)
    # Let caller handle status logic
    r.raise_for_status()
    return parse_tsv(r.text)

def fetch_uniprot_fasta(accession: str, session: requests.Session, timeout_s: int = 30) -> str:
    url = UNIPROT_FASTA_URL.format(acc=accession)
    r = session.get(url, timeout=timeout_s)
    if r.status_code == 404:
        # fallback
        r = session.get(f"https://rest.uniprot.org/uniprotkb/{accession}?format=fasta", timeout=timeout_s)
    r.raise_for_status()
    lines = r.text.splitlines()
    seq = "".join([ln.strip() for ln in lines if ln and not ln.startswith(">")])
    return seq

def load_cache(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(path: str, cache: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flamholz_csv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_missing", required=True)
    ap.add_argument("--cache_json", default="")
    ap.add_argument("--sleep_s", type=float, default=0.2)
    ap.add_argument("--timeout_s", type=int, default=30)
    ap.add_argument("--max_species", type=int, default=0, help="0 = no limit")
    ap.add_argument("--wt_only", action="store_true", default=True,
                    help="Use WT rows only (primary==1, mutant==0). Default True.")
    ap.add_argument("--include_all_rows", action="store_true",
                    help="If set, do NOT filter by WT; use all species in CSV.")
    ap.add_argument("--prefer_reviewed", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.flamholz_csv)
    if "species" not in df.columns:
        raise ValueError("Expected a 'species' column in flamholz_csv.")

    if args.include_all_rows:
        df_use = df.copy()
    else:
        # WT rows: primary==1, mutant==0 is the most direct reading of Flamholz S1 structure.
        # Also this avoids mutated species strings.
        df_use = df[(df.get("primary", 1) == 1) & (df.get("mutant", 0) == 0)].copy()

    species_list = sorted(df_use["species"].dropna().unique().tolist())
    if args.max_species and args.max_species > 0:
        species_list = species_list[: args.max_species]

    cache = load_cache(args.cache_json)

    session = requests.Session()
    # A polite UA helps some endpoints; UniProt also recommends identifying automated clients.
    session.headers.update({"User-Agent": "rubisco-embedding-pipeline/1.0 (contact: your_email@example.com)"})

    meta_rows = []
    fasta_records = []
    missing = []

    used_labels = set()

    for i, species in enumerate(species_list, start=1):
        key = f"species::{species}"
        if key in cache:
            cached = cache[key]
            if cached.get("status") == "ok":
                meta_rows.append(cached["meta"])
                fasta_records.append((cached["meta"]["label"], cached["sequence"]))
                continue
            else:
                missing.append(species)
                continue

        best = None
        best_name = None
        best_hits = None

        # Try multiple organism-name candidates
        for org_name in candidate_organism_names(species):
            try:
                hits = uniprot_search(org_name, session=session, timeout_s=args.timeout_s)
            except requests.HTTPError as e:
                # Some names will be invalid in UniProt's query language; try next candidate.
                hits = pd.DataFrame()
            except Exception:
                hits = pd.DataFrame()

            if hits is None or hits.empty:
                continue

            # Score hits
            hits = hits.copy()
            hits["_score"] = hits.apply(score_hit, axis=1)
            hits = hits.sort_values(["_score"], ascending=False)

            best = hits.iloc[0].to_dict()
            best_name = org_name
            best_hits = hits
            break

        if best is None:
            cache[key] = {"status": "missing"}
            missing.append(species)
            save_cache(args.cache_json, cache)
            continue

        accession = str(best.get("Entry", best.get("accession", ""))).strip()
        if not accession:
            cache[key] = {"status": "missing"}
            missing.append(species)
            save_cache(args.cache_json, cache)
            continue

        # Fetch sequence
        try:
            raw_seq = fetch_uniprot_fasta(accession, session=session, timeout_s=args.timeout_s)
            seq = sanitize_sequence(raw_seq)
        except Exception:
            cache[key] = {"status": "missing"}
            missing.append(species)
            save_cache(args.cache_json, cache)
            continue

        # Label for FASTA (first token must be whitespace-free for ESM parsing)
        label = slugify(species)
        if label in used_labels:
            # Make it unique
            k = 2
            while f"{label}_{k}" in used_labels:
                k += 1
            label = f"{label}_{k}"
        used_labels.add(label)

        meta = {
            "species": species,
            "label": label,
            "uniprot_accession": accession,
            "uniprot_organism_query": best_name,
            "uniprot_organism_name": str(best.get("Organism", best.get("organism_name", ""))),
            "uniprot_protein_name": str(best.get("Protein names", best.get("protein_name", ""))),
            "uniprot_gene_names": str(best.get("Gene Names", best.get("gene_names", ""))),
            "uniprot_length": int(best.get("Length", best.get("length", 0))) if str(best.get("Length", "")).isdigit() else None,
            "seq_len": len(seq),
            "seq_md5": md5_hex(seq),
        }

        meta_rows.append(meta)
        fasta_records.append((label, seq))

        cache[key] = {"status": "ok", "meta": meta, "sequence": seq}
        save_cache(args.cache_json, cache)

        time.sleep(max(0.0, args.sleep_s))

    # Write outputs
    os.makedirs(os.path.dirname(args.out_fasta), exist_ok=True)
    with open(args.out_fasta, "w", encoding="utf-8") as f:
        for label, seq in fasta_records:
            f.write(f">{label}\n")
            # wrap 60 chars
            for j in range(0, len(seq), 60):
                f.write(seq[j:j+60] + "\n")

    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    pd.DataFrame(meta_rows).to_csv(args.out_meta, index=False)

    os.makedirs(os.path.dirname(args.out_missing), exist_ok=True)
    with open(args.out_missing, "w", encoding="utf-8") as f:
        for s in missing:
            f.write(s + "\n")

    print(f"[done] species requested: {len(species_list)}")
    print(f"[done] sequences fetched: {len(fasta_records)}")
    print(f"[done] missing species: {len(missing)}")
    print(f"[done] FASTA: {args.out_fasta}")
    print(f"[done] meta:  {args.out_meta}")
    print(f"[done] missing list: {args.out_missing}")

if __name__ == "__main__":
    main()
