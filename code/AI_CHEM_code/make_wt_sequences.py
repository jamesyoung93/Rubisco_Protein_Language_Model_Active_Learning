import numpy as np
import pandas as pd

df = pd.read_csv("rubisco_datasets_merged.csv", low_memory=False)

def pick_best_offset(dms):
    # idx = position_external - 1 + offset
    offsets = [-2, -1, 0, 1, 2]
    best_off, best_frac = None, -1.0
    lengths = dms["aa_sequence"].str.len().to_numpy()

    for off in offsets:
        idx = dms["position_external"].to_numpy() - 1 + off
        ok = (idx >= 0) & (idx < lengths)
        d = dms.loc[ok].copy()
        idx_ok = idx[ok].astype(int)

        muts = d["mut_residue"].astype(str).to_numpy()
        seqs = d["aa_sequence"].astype(str).to_numpy()

        match = np.fromiter((s[i] == m for s, i, m in zip(seqs, idx_ok, muts)),
                            dtype=bool, count=len(d))
        frac = float(match.mean()) if len(match) else 0.0
        print(f"offset {off:+d}: mut-residue match fraction {frac:.3f} (n={len(match)})")

        if frac > best_frac:
            best_frac, best_off = frac, off

    if best_frac < 0.90:
        raise RuntimeError(f"Could not validate DMS indexing; best match fraction={best_frac:.3f} at offset={best_off}")
    print(f"Chosen offset: {best_off:+d} (match fraction {best_frac:.3f})")
    return best_off

# -------- DMS WT reconstruction --------
dms = df[df["dataset_id"] == "DMS"].dropna(subset=["aa_sequence","position_external","wt_residue","mut_residue"]).copy()
dms["position_external"] = pd.to_numeric(dms["position_external"], errors="coerce")
dms = dms.dropna(subset=["position_external"])
dms["position_external"] = dms["position_external"].astype(int)

offset = pick_best_offset(dms)

# pick a row where the sequence actually matches the recorded mutant residue at idx
for _, row in dms.iterrows():
    p = int(row["position_external"])
    idx = p - 1 + offset
    seq = str(row["aa_sequence"])
    if idx < 0 or idx >= len(seq):
        continue
    if seq[idx] == str(row["mut_residue"]):
        wt_seq = seq[:idx] + str(row["wt_residue"]) + seq[idx+1:]
        break
else:
    raise RuntimeError("Failed to find a DMS row consistent with mut_residue for WT reconstruction.")

# validate WT against many rows: WT residue at each position should equal wt_residue
idx_all = dms["position_external"].to_numpy() - 1 + offset
ok = (idx_all >= 0) & (idx_all < len(wt_seq))
wt_res = dms.loc[ok, "wt_residue"].astype(str).to_numpy()
idx_ok = idx_all[ok].astype(int)
match_wt = np.fromiter((wt_seq[i] == r for i, r in zip(idx_ok, wt_res)),
                       dtype=bool, count=len(idx_ok))
print(f"DMS WT validation: fraction positions matching wt_residue = {float(match_wt.mean()):.4f} (n={len(match_wt)})")

# -------- HOFF WT --------
hoff = df[df["dataset_id"] == "HOFF"].copy()
hoff["n_mut"] = pd.to_numeric(hoff["n_mut"], errors="coerce")
hoff_wt = hoff[(hoff["n_mut"] == 0) & (hoff.get("has_stop").fillna(False) == False)].dropna(subset=["aa_sequence"])

if len(hoff_wt) == 0:
    # fallback: use the most common sequence among lowest-mutation rows
    hoff_low = hoff.dropna(subset=["aa_sequence"]).copy()
    wt_seq_hoff = hoff_low["aa_sequence"].mode().iloc[0]
    print("WARNING: No n_mut==0 HOFF row found; using modal sequence as WT.")
else:
    # if multiple, ensure they agree
    seqs = hoff_wt["aa_sequence"].astype(str).unique()
    wt_seq_hoff = seqs[0]
    if len(seqs) > 1:
        print("WARNING: Multiple distinct HOFF WT sequences found; using the first unique one.")

out = pd.DataFrame([
    {"variant_id":"DMS|WT",  "dataset_id":"DMS",  "aa_sequence":wt_seq},
    {"variant_id":"HOFF|WT", "dataset_id":"HOFF", "aa_sequence":wt_seq_hoff},
])

out.to_csv("wt_sequences_for_embedding.csv", index=False)
print("Wrote wt_sequences_for_embedding.csv")
print(out[["variant_id","dataset_id"]])
