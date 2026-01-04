#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, time
from io import StringIO
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests

TAX_SEARCH = "https://rest.uniprot.org/taxonomy/search"
KB_SEARCH  = "https://rest.uniprot.org/uniprotkb/search"
KB_FASTA   = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

GENE_TERMS = ["rbcl", "cbbl", "cbbm"]  # Form I/II large-chain gene naming

def canon_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def slug(s: str) -> str:
    s = canon_ws(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:120] if s else "unknown"

def taxonomy_lookup(session: requests.Session, species: str, timeout_s: int = 30) -> Optional[Tuple[int, str]]:
    """Return (taxonId, scientificName) or None."""
    params = {"query": species, "format": "json", "size": 10}
    r = session.get(TAX_SEARCH, params=params, timeout=timeout_s)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None

    # Prefer entries whose scientificName contains the numeric token (if present)
    token = None
    m = re.search(r"(PCC\s?\d+|ATCC\s?\d+|UTEX\s?\d+|\d{4,6})", species, flags=re.I)
    if m:
        token = m.group(1).replace(" ", "").upper()

    def sci(rec): return (rec.get("scientificName") or rec.get("name") or "").strip()
    def tid(rec): return rec.get("taxonId")

    if token:
        for rec in results:
            sn = sci(rec).replace(" ", "").upper()
            if token in sn and tid(rec) is not None:
                return int(tid(rec)), sci(rec)

    # Fall back to top result
    rec = results[0]
    if tid(rec) is None:
        return None
    return int(tid(rec)), sci(rec)

def uniprot_search_rbcl(session: requests.Session, taxid: int, timeout_s: int = 30) -> pd.DataFrame:
    gene_clause = " OR ".join([f"gene:{g}" for g in GENE_TERMS])
    q = f"(organism_id:{taxid}) AND ({gene_clause})"
    params = {
        "query": q,
        "format": "tsv",
        "size": 50,
        "fields": "accession,reviewed,protein_name,organism_name,organism_id,length,gene_names",
    }
    r = session.get(KB_SEARCH, params=params, timeout=timeout_s)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(r.text), sep="\t")

def score_hit(row: pd.Series) -> float:
    score = 0.0
    reviewed = str(row.get("Reviewed", row.get("reviewed", ""))).strip().lower()
    if reviewed in {"reviewed", "true", "yes", "1"}:
        score += 5.0

    pname = str(row.get("Protein names", row.get("protein_name", ""))).lower()
    if "fragment" in pname:
        score -= 10.0
    if "ribulose" in pname and "carboxylase" in pname:
        score += 2.0

    genes = str(row.get("Gene Names", row.get("gene_names", ""))).lower()
    if any(g in genes for g in GENE_TERMS):
        score += 2.0

    try:
        L = int(row.get("Length", row.get("length", -1)))
    except Exception:
        L = -1
    # Broadly accept 300–700 aa; favor typical Form I/II large-chain sizes
    if 420 <= L <= 600:
        score += 2.0
    elif 300 <= L <= 700:
        score += 0.5
    else:
        score -= 1.0

    return score

def fetch_fasta(session: requests.Session, acc: str, timeout_s: int = 30) -> str:
    r = session.get(KB_FASTA.format(acc=acc), timeout=timeout_s)
    r.raise_for_status()
    lines = r.text.splitlines()
    seq = "".join([ln.strip() for ln in lines if ln and not ln.startswith(">")])
    return seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flamholz_csv", required=True)
    ap.add_argument("--out_fasta", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_missing", required=True)
    ap.add_argument("--cache_json", default="")
    ap.add_argument("--sleep_s", type=float, default=0.2)
    ap.add_argument("--timeout_s", type=int, default=30)
    ap.add_argument("--wt_only", action="store_true", default=True, help="Use primary==1 and mutant==0 only.")
    args = ap.parse_args()

    df = pd.read_csv(args.flamholz_csv)
    if "species" not in df.columns:
        raise SystemExit("Flamholz CSV must include a 'species' column.")

    if args.wt_only and ("primary" in df.columns) and ("mutant" in df.columns):
        df = df[(df["primary"] == 1) & (df["mutant"] == 0)].copy()

    species_list = sorted(df["species"].dropna().unique().tolist())

    cache: Dict = {}
    if args.cache_json and os.path.exists(args.cache_json):
        with open(args.cache_json, "r", encoding="utf-8") as f:
            cache = json.load(f)

    os.makedirs(os.path.dirname(args.out_fasta), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_missing), exist_ok=True)
    if args.cache_json:
        os.makedirs(os.path.dirname(args.cache_json), exist_ok=True)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "rubisco-flamholz-seqfetch/1.0"})

    meta_rows = []
    fasta_records = []
    missing = []

    for sp in species_list:
        sp_key = f"species::{sp}"
        if sp_key in cache:
            c = cache[sp_key]
            if c.get("status") == "ok":
                meta_rows.append(c["meta"])
                fasta_records.append((c["meta"]["label"], c["seq"]))
                continue
            else:
                missing.append(sp)
                continue

        try:
            tax = taxonomy_lookup(sess, sp, timeout_s=args.timeout_s)
            if tax is None:
                cache[sp_key] = {"status": "missing"}
                missing.append(sp)
                continue
            taxid, taxname = tax

            hits = uniprot_search_rbcl(sess, taxid, timeout_s=args.timeout_s)
            if hits.empty:
                cache[sp_key] = {"status": "missing"}
                missing.append(sp)
                continue

            hits["_score"] = hits.apply(score_hit, axis=1)
            hits = hits.sort_values("_score", ascending=False)

            acc = str(hits.iloc[0].get("Entry", hits.iloc[0].get("accession", ""))).strip()
            if not acc:
                cache[sp_key] = {"status": "missing"}
                missing.append(sp)
                continue

            seq = fetch_fasta(sess, acc, timeout_s=args.timeout_s)
            label = slug(sp)

            meta = {
                "species": sp,
                "label": label,
                "taxon_id": taxid,
                "taxon_name": taxname,
                "uniprot_accession": acc,
                "uniprot_protein_name": str(hits.iloc[0].get("Protein names", "")),
                "uniprot_gene_names": str(hits.iloc[0].get("Gene Names", "")),
                "uniprot_length": int(hits.iloc[0].get("Length", 0)),
                "seq_len": len(seq),
            }

            meta_rows.append(meta)
            fasta_records.append((label, seq))

            cache[sp_key] = {"status": "ok", "meta": meta, "seq": seq}

        except Exception:
            cache[sp_key] = {"status": "missing"}
            missing.append(sp)

        if args.cache_json:
            with open(args.cache_json, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        time.sleep(max(0.0, args.sleep_s))

    with open(args.out_fasta, "w", encoding="utf-8") as f:
        for label, seq in fasta_records:
            f.write(f">{label}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    pd.DataFrame(meta_rows).to_csv(args.out_meta, index=False)
    with open(args.out_missing, "w", encoding="utf-8") as f:
        for sp in missing:
            f.write(sp + "\n")

    print(f"[done] species requested: {len(species_list)}")
    print(f"[done] sequences fetched: {len(fasta_records)}")
    print(f"[done] missing species: {len(missing)}")
    print(f"[done] FASTA: {args.out_fasta}")
    print(f"[done] meta:  {args.out_meta}")
    print(f"[done] missing list: {args.out_missing}")

if __name__ == "__main__":
    main()
