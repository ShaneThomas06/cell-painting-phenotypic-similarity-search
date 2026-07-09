from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity

from cell_painting_profiling.embeddings.aggregate import get_embedding_columns


def build_compound_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "perturbation_id",
        "compound_name",
        "mechanism_of_action",
        "smiles",
        "inchikey",
        "pubchem_cid",
    }
    missing = required.difference(metadata.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Metadata table is missing required columns: {missing_text}")

    return (
        metadata[list(required)]
        .drop_duplicates()
        .sort_values(["compound_name", "perturbation_id"])
        .reset_index(drop=True)
    )


def attach_chemical_metadata(fingerprints: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    compound_metadata = build_compound_metadata(metadata)
    chemical_columns = ["perturbation_id", "smiles", "inchikey", "pubchem_cid"]
    merged = fingerprints.merge(
        compound_metadata[chemical_columns].drop_duplicates("perturbation_id"),
        on="perturbation_id",
        how="left",
    )
    return merged


def smiles_to_fingerprint(smiles: str | float | None, radius: int = 2, n_bits: int = 2048):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(molecule)


def build_pairwise_similarity_table(compounds: pd.DataFrame) -> pd.DataFrame:
    embedding_columns = get_embedding_columns(compounds)
    if not embedding_columns:
        raise ValueError("Compound table has no embedding_* columns")

    valid = compounds.loc[compounds["smiles"].fillna("").astype(str).str.len() > 0].copy()
    valid["chemical_fingerprint"] = valid["smiles"].apply(smiles_to_fingerprint)
    valid = valid.loc[valid["chemical_fingerprint"].notna()].reset_index(drop=True)
    if len(valid) < 2:
        raise ValueError("At least two compounds with valid SMILES are required")

    morphology = cosine_similarity(valid[embedding_columns].to_numpy())
    rows = []
    for left_index, right_index in combinations(range(len(valid)), 2):
        left = valid.iloc[left_index]
        right = valid.iloc[right_index]
        chemical_similarity = DataStructs.TanimotoSimilarity(
            left["chemical_fingerprint"],
            right["chemical_fingerprint"],
        )
        rows.append(
            {
                "compound_a_id": left["perturbation_id"],
                "compound_a_name": left["compound_name"],
                "compound_a_mechanism": left["mechanism_of_action"],
                "compound_b_id": right["perturbation_id"],
                "compound_b_name": right["compound_name"],
                "compound_b_mechanism": right["mechanism_of_action"],
                "same_mechanism": left["mechanism_of_action"] == right["mechanism_of_action"],
                "morphology_similarity": float(morphology[left_index, right_index]),
                "chemical_similarity": float(chemical_similarity),
            }
        )

    pairwise = pd.DataFrame(rows)
    pairwise["morphology_rank"] = pairwise["morphology_similarity"].rank(
        ascending=False,
        method="min",
    )
    pairwise["chemical_rank"] = pairwise["chemical_similarity"].rank(
        ascending=False,
        method="min",
    )
    pairwise["morphology_chemistry_gap"] = (
        pairwise["morphology_similarity"] - pairwise["chemical_similarity"]
    )
    return pairwise.sort_values(
        ["morphology_similarity", "chemical_similarity"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_case_studies(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise.copy()

    cases = []
    definitions = [
        (
            "morphologically similar, chemically dissimilar",
            pairwise.sort_values("morphology_chemistry_gap", ascending=False),
        ),
        (
            "chemically similar, morphologically dissimilar",
            pairwise.sort_values("morphology_chemistry_gap"),
        ),
        (
            "same mechanism, high morphology",
            pairwise.loc[pairwise["same_mechanism"]].sort_values(
                "morphology_similarity",
                ascending=False,
            ),
        ),
        (
            "same mechanism, low morphology",
            pairwise.loc[pairwise["same_mechanism"]].sort_values(
                "morphology_similarity",
                ascending=True,
            ),
        ),
    ]
    for case_type, frame in definitions:
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["case_type"] = case_type
        cases.append(row)
    return pd.DataFrame(cases)


def plot_morphology_chemistry(pairwise: pd.DataFrame, output_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=pairwise,
        x="chemical_similarity",
        y="morphology_similarity",
        hue="same_mechanism",
        style="same_mechanism",
        s=70,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
    )
    ax.set_xlabel("Chemical similarity, Morgan fingerprint Tanimoto")
    ax.set_ylabel("Morphology similarity, CNN embedding cosine")
    ax.set_title("Morphology and chemistry similarity across compound pairs")
    ax.legend(title="Shared mechanism")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_analysis(
    fingerprints_path: str | Path,
    metadata_path: str | Path,
    output_pairwise: str | Path,
    output_cases: str | Path,
    output_figure: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fingerprints = pd.read_csv(fingerprints_path)
    metadata = pd.read_csv(metadata_path)
    compounds = attach_chemical_metadata(fingerprints, metadata)
    pairwise = build_pairwise_similarity_table(compounds)
    cases = select_case_studies(pairwise)

    output_pairwise = Path(output_pairwise)
    output_pairwise.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_pairwise, index=False)

    output_cases = Path(output_cases)
    output_cases.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_cases, index=False)

    plot_morphology_chemistry(pairwise, output_figure)
    return pairwise, cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Cell Painting morphology and chemical similarity.")
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-pairwise", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-figure", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairwise, cases = run_analysis(
        fingerprints_path=args.fingerprints,
        metadata_path=args.metadata,
        output_pairwise=args.output_pairwise,
        output_cases=args.output_cases,
        output_figure=args.output_figure,
    )
    print(f"Wrote {len(pairwise)} compound pairs")
    print(cases[["case_type", "compound_a_name", "compound_b_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
