import pandas as pd

from cell_painting_profiling.analysis.morphology_chemistry import (
    attach_chemical_metadata,
    build_case_study_summary,
    build_pairwise_similarity_table,
    select_case_studies,
    smiles_to_fingerprint,
)


def test_smiles_to_fingerprint_handles_valid_and_missing_smiles():
    assert smiles_to_fingerprint("CCO") is not None
    assert smiles_to_fingerprint("") is None
    assert smiles_to_fingerprint(None) is None


def test_pairwise_table_computes_morphology_and_chemistry_similarity():
    compounds = pd.DataFrame(
        {
            "perturbation_id": ["a", "b", "c"],
            "compound_name": ["A", "B", "C"],
            "mechanism_of_action": ["m1", "m1", "m2"],
            "smiles": ["CCO", "CCN", "c1ccccc1"],
            "embedding_0000": [1.0, 0.9, 0.0],
            "embedding_0001": [0.0, 0.1, 1.0],
        }
    )

    pairwise = build_pairwise_similarity_table(compounds)

    assert len(pairwise) == 3
    assert set(pairwise["same_mechanism"]) == {True, False}
    assert pairwise["morphology_similarity"].between(0, 1).all()
    assert pairwise["chemical_similarity"].between(0, 1).all()
    assert "morphology_chemistry_gap" in pairwise.columns


def test_select_case_studies_returns_named_cases():
    pairwise = pd.DataFrame(
        {
            "compound_a_name": ["A", "A", "B", "C"],
            "compound_b_name": ["B", "C", "D", "D"],
            "same_mechanism": [True, False, False, True],
            "morphology_similarity": [0.95, 0.9, 0.1, 0.2],
            "chemical_similarity": [0.9, 0.1, 0.95, 0.2],
            "morphology_chemistry_gap": [0.05, 0.8, -0.85, 0.0],
        }
    )

    cases = select_case_studies(pairwise)

    assert not cases.empty
    assert "case_type" in cases.columns
    assert "morphologically similar, chemically dissimilar" in set(cases["case_type"])
    assert "chemically similar, morphologically dissimilar" in set(cases["case_type"])



def test_build_case_study_summary_combines_duplicate_pairs():
    cases = pd.DataFrame(
        {
            "compound_a_id": ["a", "a", "c"],
            "compound_a_name": ["A", "A", "C"],
            "compound_a_mechanism": ["m1", "m1", "m2"],
            "compound_b_id": ["b", "b", "d"],
            "compound_b_name": ["B", "B", "D"],
            "compound_b_mechanism": ["m1", "m1", "m3"],
            "same_mechanism": [True, True, False],
            "morphology_similarity": [0.9, 0.9, 0.2],
            "chemical_similarity": [0.1, 0.1, 0.8],
            "morphology_rank": [1.0, 1.0, 3.0],
            "chemical_rank": [3.0, 3.0, 1.0],
            "morphology_chemistry_gap": [0.8, 0.8, -0.6],
            "case_type": ["case one", "case two", "case three"],
        }
    )

    summary = build_case_study_summary(cases)

    assert len(summary) == 2
    assert summary.iloc[0]["case_type"] == "case one; case two"

def test_attach_chemical_metadata_joins_by_perturbation_id():
    fingerprints = pd.DataFrame(
        {
            "perturbation_id": ["a"],
            "compound_name": ["A"],
            "mechanism_of_action": ["m1"],
            "embedding_0000": [1.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "perturbation_id": ["a"],
            "compound_name": ["A"],
            "mechanism_of_action": ["m1"],
            "smiles": ["CCO"],
            "inchikey": ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
            "pubchem_cid": ["702"],
        }
    )

    merged = attach_chemical_metadata(fingerprints, metadata)

    assert merged.iloc[0]["smiles"] == "CCO"
