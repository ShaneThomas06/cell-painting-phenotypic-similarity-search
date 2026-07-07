import pandas as pd

from cell_painting_profiling.data.normalize_cpg0002 import normalize_cpg0002_metadata


def test_normalize_cpg0002_metadata_joins_well_to_moa():
    load_data = pd.DataFrame(
        {
            "URL_OrigBrightField": ["s3://bf"],
            "URL_OrigRNA": ["s3://rna"],
            "URL_OrigMito": ["s3://mito"],
            "URL_OrigAGP": ["s3://agp"],
            "URL_OrigER": ["s3://er"],
            "URL_OrigDNA": ["s3://dna"],
            "Metadata_Well": ["A01"],
            "Metadata_Site": [1],
            "Metadata_Plate": ["plate_1"],
        }
    )
    platemap = pd.DataFrame(
        {
            "well_position": ["A01"],
            "broad_sample": ["BRD-A"],
            "solvent": ["DMSO"],
        }
    )
    compound_metadata = pd.DataFrame(
        {
            "broad_sample": ["BRD-A"],
            "InChIKey": ["AAA"],
            "pert_iname": ["compound_a"],
            "pubchem_cid": [1],
            "moa": ["HDAC inhibitor"],
            "pert_type": ["trt"],
            "control_type": [None],
            "smiles": ["CCO"],
        }
    )

    normalized = normalize_cpg0002_metadata(load_data, platemap, compound_metadata)

    assert normalized.loc[0, "perturbation_id"] == "BRD-A"
    assert normalized.loc[0, "compound_name"] == "compound_a"
    assert normalized.loc[0, "mechanism_of_action"] == "HDAC inhibitor"
    assert normalized.loc[0, "url_dna"] == "s3://dna"
    assert normalized.loc[0, "image_record_id"] == "cpg0002-jump-scope__plate_1__A01__site1"
