"""
Fetch SMILES for all drugs in the dataset from PubChem.
Saves results to data/drug_smiles.csv.

Usage:
    python scripts/fetch_smiles.py
"""

import time
import urllib.parse
import urllib.request
import json
import argparse
import pandas as pd

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"


def fetch_smiles_pubchem(name: str) -> str | None:
    url = PUBCHEM_URL.format(urllib.parse.quote(name))
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            props = data["PropertyTable"]["Properties"][0]
            return props.get("IsomericSMILES") or props.get("SMILES")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccle", default="data/ccle_filtered_merged_df.parquet")
    parser.add_argument("--patient", default="data/filtered_merged_df.parquet")
    parser.add_argument("--out", default="data/drug_smiles.csv")
    parser.add_argument("--delay", type=float, default=0.03, help="seconds between PubChem requests")
    args = parser.parse_args()

    ccle_df = pd.read_parquet(args.ccle)
    pat_df = pd.read_parquet(args.patient)
    drugs = sorted(set(ccle_df["DRUG_NAME"]) | set(pat_df["DRUG_NAME"]))
    print(f"Fetching SMILES for {len(drugs)} drugs from PubChem...")

    results = []
    for i, name in enumerate(drugs, 1):
        smiles = fetch_smiles_pubchem(name)
        status = "ok" if smiles else "not found"
        results.append({"DRUG_NAME": name, "SMILES": smiles})
        print(f"[{i:3d}/{len(drugs)}] {name}: {status}")
        time.sleep(args.delay)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    found = df["SMILES"].notna().sum()
    print(f"\nDone: {found}/{len(drugs)} SMILES found → {args.out}")
    if found < len(drugs):
        missing = df[df["SMILES"].isna()]["DRUG_NAME"].tolist()
        print(f"Missing ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
