# Clinical_Frequency.py

import os
import numpy as np
import pandas as pd
import requests
import time
import re
import json
import random
import sqlite3

from datetime import datetime, timezone

# ============================================================
# Analysis settings
# ============================================================

INCLUDE_LIKELY_PATHOGENIC = False

# Minimum gnomAD allele number required for a population estimate.
MIN_POPULATION_AN = 10_000

# Maximum AF allowed for the rare, highly penetrant disease model.
# Variants at or above this frequency are retained for QC,
# but excluded from prevalence calculations.
MAX_MODEL_ELIGIBLE_AF = 0.01

EXCLUDED_POPULATION_IDS = {
    "remaining",
    "remaining_XX",
    "remaining_XY",
}

# ============================================================
# GRCh38 pseudoautosomal regions
# ============================================================

GRCH38_PAR_INTERVALS = [
    (10_001, 2_781_479),
    (155_701_383, 156_030_895),
]

# ============================================================
# Validity display settings
# ============================================================

VALIDITY_RANK = {
    "Definitive": 8,
    "Strong": 7,
    "Moderate": 6,
    "Supportive": 5,
    "Limited": 4,
    "Animal Model Only": 3,
    "Disputed": 2,
    "Disputed Evidence": 2,
    "Refuted": 1,
    "Refuted Evidence": 1,
    "No Known Disease Relationship": 0,
    "No supporting assertion": -1,
}

RANK_TO_VALIDITY = {
    8: "Definitive",
    7: "Strong",
    6: "Moderate",
    5: "Supportive",
    4: "Limited",
    3: "Animal Model Only",
    2: "Disputed",
    1: "Refuted",
    0: "No Known Disease Relationship",
    -1: "No supporting assertion",
}

# ============================================================
# Variant evidence status labels
# ============================================================

STATUS_INCLUDED = "included"
STATUS_GNOMAD_NOT_FOUND = "gnomad_not_found"
STATUS_INSUFFICIENT_AN = "insufficient_population_an"
STATUS_EXCLUDED_COMMON = "excluded_common"
STATUS_QUERY_FAILED = "query_failed"

# ============================================================
# gnomAD cache settings
# ============================================================

# For development only. Later, set this to None for all variants.
GNOMAD_TEST_LIMIT = 100

GNOMAD_CACHE_FILE = "gnomad_cache.sqlite"
GNOMAD_DATASET = "gnomad_r4"

# Refresh cached gnomAD results after this many days.
GNOMAD_CACHE_MAX_AGE_DAYS = 90

# Set True to ignore cache age and query gnomAD again.
FORCE_GNOMAD_REFRESH = False

# ============================================================
# ClinGen inheritance settings
# ============================================================

CLINGEN_FILE = "clingen_gene_disease_validity.csv"
CLINGEN_URL = (
    "https://search.clinicalgenome.org/"
    "kb/gene-validity/download"
)
CLINGEN_CACHE_MAX_AGE_DAYS = 7


# ============================================================
# GenCC inheritance settings
# ============================================================

GENCC_FILE = "gencc_submissions.csv"
GENCC_URL = (
    "https://thegencc.org/download/action/"
    "submissions-export-csv?format=new"
)
GENCC_CACHE_MAX_AGE_DAYS = 7
GENCC_METADATA_FILE = "gencc_download_metadata.json"

X_LINKED_MODELS = {
    "x_linked_recessive",
    "x_linked_dominant",
    "x_linked_unspecified",
}

# ------------------------------------------------------------
# Download latest ClinVar data
# ------------------------------------------------------------

clinvar_url = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/"
    "tab_delimited/variant_summary.txt.gz"
)
output_file = "variation_summary.txt.gz"

MAX_FILE_AGE_DAYS = 7


def file_age_days(filepath):
    """Return the age of a file in days."""
    modified_time = os.path.getmtime(filepath)
    current_time = time.time()

    return (current_time - modified_time) / (60 * 60 * 24)

def download_file(url, filepath, timeout=120):
    """
    Safely download a file, replacing the existing copy only
    after the complete download succeeds.
    """

    print(f"Downloading {filepath}...")

    response = requests.get(
        url,
        stream=True,
        timeout=timeout,
    )

    response.raise_for_status()

    temp_file = filepath + ".tmp"

    try:
        with open(temp_file, "wb") as file_handle:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file_handle.write(chunk)

        os.replace(
            temp_file,
            filepath,
        )

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(
        f"Download complete: {filepath}"
    )

def ensure_clingen_file():
    """
    Ensure a reasonably recent ClinGen Gene-Disease Validity
    CSV is available locally.
    """

    if os.path.exists(CLINGEN_FILE):

        age_days = file_age_days(
            CLINGEN_FILE
        )

        print(
            f"Existing ClinGen file is "
            f"{age_days:.1f} days old."
        )

        if age_days < CLINGEN_CACHE_MAX_AGE_DAYS:
            print(
                "Using existing ClinGen file."
            )
            return

        print(
            "ClinGen file is at least "
            f"{CLINGEN_CACHE_MAX_AGE_DAYS} "
            "days old. Updating..."
        )

    else:

        print(
            "No existing ClinGen file found."
        )

    download_file(
        CLINGEN_URL,
        CLINGEN_FILE,
    )

def load_gencc_metadata():
    """
    Load saved GenCC download metadata.
    """

    if not os.path.exists(
        GENCC_METADATA_FILE
    ):
        return {}

    try:
        with open(
            GENCC_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file_handle:
            return json.load(file_handle)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

def save_gencc_metadata(metadata):
    """
    Save GenCC download metadata.
    """

    with open(
        GENCC_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            metadata,
            file_handle,
            indent=2,
        )

def ensure_gencc_file():
    """
    Ensure a current GenCC new-format submissions CSV
    is available locally.

    Uses the server ETag when possible so unchanged files
    do not need to be downloaded again.
    """

    metadata = load_gencc_metadata()

    if os.path.exists(GENCC_FILE):

        downloaded_at = metadata.get(
            "downloaded_at"
        )

        if downloaded_at:

            try:
                downloaded_time = (
                    datetime.fromisoformat(
                        downloaded_at
                    )
                )

                if downloaded_time.tzinfo is None:
                    downloaded_time = (
                        downloaded_time.replace(
                            tzinfo=timezone.utc
                        )
                    )

                age = (
                    datetime.now(timezone.utc)
                    - downloaded_time
                )

                if age.total_seconds() < (
                    GENCC_CACHE_MAX_AGE_DAYS
                    * 24
                    * 60
                    * 60
                ):

                    print(
                        "Using existing GenCC file."
                    )

                    return

            except ValueError:
                pass

    headers = {}

    saved_etag = metadata.get("etag")

    if saved_etag:
        headers["If-None-Match"] = saved_etag

    # If there is no local file, force a normal download.
    if not os.path.exists(GENCC_FILE):
        headers = {}

    print(
        "Checking GenCC reference data..."
    )

    response = requests.get(
        GENCC_URL,
        headers=headers,
        stream=True,
        timeout=120,
    )

    # GenCC explicitly uses HTTP 304 when our ETag
    # matches the current release.
    if response.status_code == 304:

        metadata[
            "downloaded_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        save_gencc_metadata(
            metadata
        )

        print(
            "GenCC data have not changed. "
            "Using existing file."
        )

        return

    response.raise_for_status()

    temp_file = GENCC_FILE + ".tmp"

    try:

        with open(
            temp_file,
            "wb",
        ) as file_handle:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file_handle.write(chunk)

        os.replace(
            temp_file,
            GENCC_FILE,
        )

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)

    new_metadata = {
        "etag": response.headers.get(
            "ETag"
        ),
        "last_modified": response.headers.get(
            "Last-Modified"
        ),
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_gencc_metadata(
        new_metadata
    )

    print(
        "GenCC reference data updated."
    )

def cache_is_current(
    cached_status,
    max_age_days=GNOMAD_CACHE_MAX_AGE_DAYS,
):
    """
    Return True if a cached gnomAD result is still considered current.
    """

    if cached_status is None:
        return False

    queried_at = cached_status.get("queried_at")

    if not queried_at:
        return False

    queried_time = datetime.fromisoformat(queried_at)

    if queried_time.tzinfo is None:
        queried_time = queried_time.replace(
            tzinfo=timezone.utc
        )

    age = datetime.now(timezone.utc) - queried_time

    return age.total_seconds() < (
        max_age_days * 24 * 60 * 60
    )

def download_clinvar(url, filepath):
    """Download the ClinVar variant summary file."""

    print("Downloading ClinVar variant summary...")

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )
    response.raise_for_status()

    # Download to a temporary file first
    temp_file = filepath + ".tmp"

    try:
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        # Only replace the old file after a successful download
        os.replace(temp_file, filepath)

    finally:
        # Clean up an incomplete temporary download if necessary
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("Download complete.")

def is_grch38_par(position):
    """
    Return True when an X-chromosome GRCh38 position lies
    within PAR1 or PAR2.
    """

    if pd.isna(position):
        return pd.NA

    position = int(position)

    return any(
        start <= position <= end
        for start, end in GRCH38_PAR_INTERVALS
    )

if os.path.exists(output_file):

    age_days = file_age_days(output_file)

    print(f"Existing ClinVar file is {age_days:.1f} days old.")

    if age_days < MAX_FILE_AGE_DAYS:
        print("Using existing ClinVar file.")

    else:
        print("ClinVar file is at least 7 days old. Updating...")
        download_clinvar(clinvar_url, output_file)

else:

    print("No existing ClinVar file found.")
    download_clinvar(clinvar_url, output_file)

# ------------------------------------------------------------
# Read ClinVar data
# ------------------------------------------------------------

print("Reading ClinVar data...")

clinvar = pd.read_csv(
    output_file,
    sep="\t",
    compression="gzip",
    low_memory=False
)

clinvar = clinvar.convert_dtypes()

print(f"Loaded {len(clinvar):,} ClinVar rows.")

## Trim for efficiency
pathovar = clinvar[
    clinvar['Assembly'].eq('GRCh38')
].copy()

pathovar = pathovar.drop(['#AlleleID', 
                    'Type',
                    'Name',
                    'nsv/esv (dbVar)',
                    'RCVaccession',
                    'Origin',
                    'Start',
                    'Stop',
                    'ReferenceAllele',
                    'AlternateAllele',
                    'Cytogenetic',
                    'ChromosomeAccession',
                    'Guidelines',
                    'OtherIDs',
                    'ClinSigSimple',
                    'SubmitterCategories',
                    'SCVsForAggregateGermlineClassification',
                    'SCVsForAggregateSomaticClinicalImpact',
                    'SCVsForAggregateOncogenicityClassification', 
                    'TestedInGTR'], axis=1)


## Create new lines for each item, and filter

pathovar['GeneSymbol'] = pathovar['GeneSymbol'].str.split(';')
pathovar = pathovar.explode('GeneSymbol')


# ============================================================
# Split ClinVar phenotype names and identifiers together
# ============================================================

pathovar["PhenotypeList"] = (
    pathovar["PhenotypeList"]
    .astype("string")
    .str.split("|")
)

pathovar["PhenotypeIDS"] = (
    pathovar["PhenotypeIDS"]
    .astype("string")
    .str.split("|")
)

# Explode the two corresponding ClinVar fields together so that
# each phenotype remains paired with its own disease identifiers.
pathovar = pathovar.explode(
    [
        "PhenotypeList",
        "PhenotypeIDS",
    ]
)

pathovar["PhenotypeList"] = (
    pathovar["PhenotypeList"]
    .astype("string")
    .str.strip()
)

pathovar["PhenotypeIDS"] = (
    pathovar["PhenotypeIDS"]
    .astype("string")
    .str.strip()
)

def extract_identifier(text, prefix):
    """
    Extract the first identifier for a given namespace
    from an aligned ClinVar PhenotypeIDS group.
    """

    if pd.isna(text):
        return pd.NA

    text = str(text)

    if prefix == "MONDO":
        match = re.search(
            r"MONDO:(?:MONDO:)?(\d+)",
            text,
        )

    elif prefix == "OMIM":
        match = re.search(
            r"OMIM:(?:PS)?(\d+)",
            text,
        )

    elif prefix == "MedGen":
        match = re.search(
            r"MedGen:(C\d+)",
            text,
        )

    elif prefix == "Orphanet":
        match = re.search(
            r"Orphanet:(\d+)",
            text,
        )

    else:
        return pd.NA

    if match is None:
        return pd.NA

    identifier = match.group(1)

    return f"{prefix}:{identifier}"

pathovar["MONDO_ID"] = pathovar["PhenotypeIDS"].apply(
    lambda x: extract_identifier(x, "MONDO")
)

pathovar["OMIM_ID"] = pathovar["PhenotypeIDS"].apply(
    lambda x: extract_identifier(x, "OMIM")
)

pathovar["MedGen_ID"] = pathovar["PhenotypeIDS"].apply(
    lambda x: extract_identifier(x, "MedGen")
)

pathovar["Orphanet_ID"] = pathovar["PhenotypeIDS"].apply(
    lambda x: extract_identifier(x, "Orphanet")
)

def choose_disease_id(row):
    """
    Choose the preferred stable disease identifier.

    Priority:
        MONDO > OMIM > Orphanet > MedGen
    """

    for column in [
        "MONDO_ID",
        "OMIM_ID",
        "Orphanet_ID",
        "MedGen_ID",
    ]:
        value = row[column]

        if pd.notna(value):
            return value

    return pd.NA


pathovar["Disease_ID"] = pathovar.apply(
    choose_disease_id,
    axis=1,
)

def normalize_hgnc_id(value):
    """
    Normalize ClinVar HGNC identifiers.
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    match = re.search(
        r"HGNC:(\d+)",
        value,
    )

    if match is None:
        return pd.NA

    return f"HGNC:{match.group(1)}"


pathovar["HGNC_ID"] = (
    pathovar["HGNC_ID"]
    .apply(normalize_hgnc_id)
)

pathovar['ClinicalSignificance'] = pathovar['ClinicalSignificance'].str.split('/')
pathovar = pathovar.explode('ClinicalSignificance')

pathovar['ClinicalSignificance'] = pathovar['ClinicalSignificance'].str.split(';')
pathovar = pathovar.explode('ClinicalSignificance')

# Clean significance labels after exploding combined classifications
pathovar['ClinicalSignificance'] = (
    pathovar['ClinicalSignificance']
    .astype('string')
    .str.strip()
)

### Only takes pathogenic variants, need a second version for including likely pathogenic
if INCLUDE_LIKELY_PATHOGENIC:
    accepted_significance = {
        'Pathogenic',
        'Likely pathogenic'
    }
else:
    accepted_significance = {
        'Pathogenic'
    }

pathovar = pathovar[
    pathovar['ClinicalSignificance'].isin(accepted_significance)
].copy()

print(
    "Included clinical significance values:",
    sorted(pathovar['ClinicalSignificance'].dropna().unique())
)

print("Rows before removing duplicates:", len(pathovar))

pathovar = pathovar.drop_duplicates(
    subset=[
        'GeneSymbol',
        'PhenotypeList',
        'Chromosome',
        'PositionVCF',
        'ReferenceAlleleVCF',
        'AlternateAlleleVCF'
    ]
)

print("Rows after removing duplicates:", len(pathovar))

print(
    "Unique variants:",
    pathovar[
        [
            'Chromosome',
            'PositionVCF',
            'ReferenceAlleleVCF',
            'AlternateAlleleVCF'
        ]
    ].drop_duplicates().shape[0]
)

pathovar = pathovar[
    ~pathovar["PhenotypeList"].str.contains(
        r"conditions|See cases|not provided|not specified",
        case=False,
        na=False,
    )
].copy()

pathovar = pathovar.reset_index(drop=True)

pathovar.to_csv("Pathogenic_Variants_temp.csv", index=False)

## Create Variant List from ClinVar Data

pvars = pd.read_csv("Pathogenic_Variants_temp.csv", delimiter=",")

pvars["PositionVCF"] = pd.to_numeric(
    pvars["PositionVCF"],
    errors="coerce",
)

# ============================================================
# Exclude ClinVar rows without a usable GRCh38 VCF allele
# ============================================================

valid_vcf_mask = (
    pvars["PositionVCF"].notna()
    & (pvars["PositionVCF"] > 0)
    & pvars["ReferenceAlleleVCF"].notna()
    & pvars["AlternateAlleleVCF"].notna()
)

invalid_vcf_rows = pvars[
    ~valid_vcf_mask
].copy()

print(
    "\nClinVar relationships without a usable VCF allele:",
    len(invalid_vcf_rows)
)

pvars = pvars[
    valid_vcf_mask
].copy()

pvars['variant'] = pvars['Chromosome'].astype(str) + '-' + pvars['PositionVCF'].astype(str) + '-' + pvars['ReferenceAlleleVCF'] + '-' + pvars['AlternateAlleleVCF']

pvars.to_csv("Pathogenic_Variants.csv", index=False)

# ============================================================
# Load ClinGen Gene-Disease Validity data
# ============================================================

ensure_clingen_file()

clingen = pd.read_csv(
    CLINGEN_FILE,
    skiprows=4,
    low_memory=False,
)

clingen = clingen[
    clingen["GENE SYMBOL"] != "+++++++++++"
].copy()

clingen = clingen[
    [
        "GENE SYMBOL",
        "GENE ID (HGNC)",
        "DISEASE LABEL",
        "DISEASE ID (MONDO)",
        "MOI",
        "CLASSIFICATION",
        "CLASSIFICATION DATE",
        "GCEP",
    ]
].copy()

clingen = clingen.rename(
    columns={
        "GENE SYMBOL": "clingen_gene_symbol",
        "GENE ID (HGNC)": "HGNC_ID",
        "DISEASE LABEL": "clingen_disease_label",
        "DISEASE ID (MONDO)": "MONDO_ID",
        "MOI": "inheritance_mode",
        "CLASSIFICATION": "gene_disease_validity",
        "CLASSIFICATION DATE": "clingen_classification_date",
        "GCEP": "clingen_gcep",
    }
)

clingen["HGNC_ID"] = (
    clingen["HGNC_ID"]
    .astype("string")
    .str.strip()
)

clingen["MONDO_ID"] = (
    clingen["MONDO_ID"]
    .astype("string")
    .str.strip()
)

clingen["inheritance_mode"] = (
    clingen["inheritance_mode"]
    .astype("string")
    .str.strip()
)

print("\nClinGen inheritance modes")
print("-------------------------")

print(
    clingen["inheritance_mode"]
    .value_counts(dropna=False)
    .to_string()
)

clingen_duplicate_keys = clingen[
    clingen.duplicated(
        subset=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        keep=False,
    )
].copy()

print(
    "\nClinGen rows with duplicated HGNC + MONDO keys:",
    len(clingen_duplicate_keys)
)

# ============================================================
# Inheritance helper functions
# ============================================================

def combine_unique_values(series):
    """
    Combine unique non-null values into a pipe-separated string.
    """

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not values:
        return pd.NA

    return "|".join(sorted(values))


def normalize_inheritance_model(label):
    """
    Normalize inheritance descriptions into model categories.
    """

    if pd.isna(label):
        return "unknown"

    label = str(label).strip().lower()

    if "autosomal recessive" in label:
        return "autosomal_recessive"

    if "autosomal dominant" in label:
        return "autosomal_dominant"

    if "x-linked recessive" in label or "x linked recessive" in label:
        return "x_linked_recessive"

    if "x-linked dominant" in label or "x linked dominant" in label:
        return "x_linked_dominant"

    if "x-linked" in label or "x linked" in label:
        return "x_linked_unspecified"

    if "y-linked" in label or "y linked" in label:
        return "y_linked"

    if "mitochondrial" in label:
        return "mitochondrial"

    if "semidominant" in label or "semi-dominant" in label:
        return "semidominant"

    if "digenic" in label:
        return "digenic"

    if "undetermined" in label:
        return "unknown"

    if label == "unknown":
        return "unknown"

    return "other"


CLINGEN_MOI_LABELS = {
    "AR": "Autosomal recessive",
    "AD": "Autosomal dominant",
    "XL": "X-linked",
    "SD": "Semidominant",
    "MT": "Mitochondrial",
    "UD": "Undetermined",
}

# ============================================================
# Build unique ClinVar gene-disease relationships
# ============================================================

clinvar_disease_relationships = (
    pvars[
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
        ]
    ]
    .drop_duplicates()
    .copy()
)

# ============================================================
# Match ClinVar relationships to direct ClinGen assertions
# ============================================================

clinvar_clingen = (
    clinvar_disease_relationships
    .merge(
        clingen,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="left",
        validate="m:m",
    )
)

clinvar_clingen["clingen_match"] = (
    clinvar_clingen[
        "inheritance_mode"
    ].notna()
)

inheritance_assertions = (
    clinvar_clingen[
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "clingen_disease_label",
            "inheritance_mode",
            "gene_disease_validity",
            "clingen_classification_date",
            "clingen_gcep",
        ]
    ]
    .dropna(
        subset=["inheritance_mode"]
    )
    .drop_duplicates()
    .copy()
)

inheritance_assertions[
    "inheritance_label"
] = (
    inheritance_assertions[
        "inheritance_mode"
    ]
    .map(CLINGEN_MOI_LABELS)
    .fillna(
        inheritance_assertions[
            "inheritance_mode"
        ]
    )
)

inheritance_assertions[
    "inheritance_model"
] = (
    inheritance_assertions[
        "inheritance_label"
    ]
    .apply(normalize_inheritance_model)
)

inheritance_assertions[
    "validity_rank"
] = (
    inheritance_assertions[
        "gene_disease_validity"
    ]
    .map(VALIDITY_RANK)
)

inheritance_assertions[
    "inheritance_source"
] = "ClinGen"

inheritance_assertions[
    "inheritance_submitter"
] = "ClinGen"

clingen_evidence = inheritance_assertions[
    [
        "GeneSymbol",
        "HGNC_ID",
        "PhenotypeList",
        "MONDO_ID",
        "inheritance_label",
        "inheritance_model",
        "gene_disease_validity",
        "validity_rank",
        "inheritance_source",
        "inheritance_submitter",
        "clingen_classification_date",
    ]
].copy()

clingen_evidence = clingen_evidence.rename(
    columns={
        "clingen_classification_date":
            "inheritance_date",
    }
)

clingen_evidence[
    "source_record_id"
] = pd.NA

relationship_match_summary = (
    clinvar_clingen[
        [
            "HGNC_ID",
            "MONDO_ID",
            "clingen_match",
        ]
    ]
    .drop_duplicates()
)

print("\nMatched ClinGen assertions by validity")
print("--------------------------------------")

print(
    inheritance_assertions[
        "gene_disease_validity"
    ]
    .value_counts(dropna=False)
    .to_string()
)

print("\nClinVar -> ClinGen inheritance matching")
print("---------------------------------------")

print(
    "Unique HGNC + MONDO relationships:",
    len(relationship_match_summary)
)

print(
    "Relationships with ClinGen inheritance:",
    relationship_match_summary[
        "clingen_match"
    ].sum()
)

print(
    "Relationships without ClinGen inheritance:",
    (
        ~relationship_match_summary[
            "clingen_match"
        ]
    ).sum()
)

# ============================================================
# Load GenCC inheritance data
# ============================================================

ensure_gencc_file()

gencc = pd.read_csv(
    GENCC_FILE,
    low_memory=False,
)

print(
    "\nGenCC rows:",
    f"{len(gencc):,}"
)

gencc_assertions = (
    gencc[
        [
            "sgc_id",
            "version_number",
            "gene_curie",
            "gene_symbol",
            "disease_curie",
            "disease_title",
            "classification_title",
            "moi_curie",
            "moi_title",
            "submitter_title",
            "submitted_as_date",
            "submitted_as_public_report_url",
        ]
    ]
    .copy()
)

gencc_assertions = gencc_assertions.rename(
    columns={
        "gene_curie": "HGNC_ID",
        "gene_symbol": "gencc_gene_symbol",
        "disease_curie": "MONDO_ID",
        "disease_title": "gencc_disease_label",
        "classification_title":
            "gene_disease_validity",
        "moi_title": "inheritance_label",
        "submitter_title":
            "inheritance_submitter",
        "submitted_as_date": "inheritance_date",
        "submitted_as_public_report_url":
            "inheritance_report_url",
    }
)

gencc_assertions[
    "inheritance_source"
] = "GenCC"

for column in [
    "HGNC_ID",
    "MONDO_ID",
    "inheritance_label",
    "gene_disease_validity",
    "inheritance_submitter",
]:
    gencc_assertions[column] = (
        gencc_assertions[column]
        .astype("string")
        .str.strip()
    )

gencc_assertions = (
    gencc_assertions
    .dropna(
        subset=[
            "HGNC_ID",
            "MONDO_ID",
        ]
    )
    .copy()
)

gencc_assertions[
    "version_number"
] = pd.to_numeric(
    gencc_assertions[
        "version_number"
    ],
    errors="coerce",
)

gencc_assertions = (
    gencc_assertions
    .sort_values(
        [
            "sgc_id",
            "version_number",
        ]
    )
    .drop_duplicates(
        subset=["sgc_id"],
        keep="last",
    )
    .reset_index(drop=True)
)

gencc_assertions[
    "inheritance_model"
] = (
    gencc_assertions[
        "inheritance_label"
    ]
    .apply(normalize_inheritance_model)
)

gencc_assertions[
    "validity_rank"
] = (
    gencc_assertions[
        "gene_disease_validity"
    ]
    .map(VALIDITY_RANK)
)

print("\nGenCC inheritance models")
print("------------------------")

print(
    gencc_assertions[
        "inheritance_model"
    ]
    .value_counts(dropna=False)
    .to_string()
)

gencc_non_clingen = gencc_assertions[
    gencc_assertions[
        "inheritance_submitter"
    ] != "ClinGen"
].copy()

print(
    "\nDirect ClinGen assertions:",
    len(clingen_evidence)
)

print(
    "GenCC assertions excluding ClinGen:",
    len(gencc_non_clingen)
)

gencc_evidence = gencc_non_clingen[
    [
        "sgc_id",
        "HGNC_ID",
        "MONDO_ID",
        "inheritance_label",
        "inheritance_model",
        "gene_disease_validity",
        "validity_rank",
        "inheritance_source",
        "inheritance_submitter",
        "inheritance_date",
    ]
].copy()

gencc_evidence[
    "source_record_id"
] = gencc_evidence[
    "sgc_id"
]

gencc_evidence = (
    clinvar_disease_relationships[
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "MONDO_ID",
        ]
    ]
    .merge(
        gencc_evidence,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="inner",
        validate="m:m",
    )
)

gencc_evidence = gencc_evidence.drop(
    columns=["sgc_id"]
)

clinvar_gencc = (
    clinvar_disease_relationships
    .merge(
        gencc_assertions[
            [
                "HGNC_ID",
                "MONDO_ID",
                "sgc_id",
            ]
        ],
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="left",
        validate="m:m",
    )
)

# ============================================================
# Calculate GenCC coverage of ClinVar relationships
# ============================================================

gencc_relationship_coverage = (
    clinvar_gencc[
        [
            "HGNC_ID",
            "MONDO_ID",
            "sgc_id",
        ]
    ]
    .groupby(
        [
            "HGNC_ID",
            "MONDO_ID",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        gencc_assertions=(
            "sgc_id",
            "count",
        )
    )
)

gencc_relationship_coverage[
    "gencc_match"
] = (
    gencc_relationship_coverage[
        "gencc_assertions"
    ] > 0
)

print("\nClinVar -> GenCC inheritance matching")
print("-------------------------------------")

print(
    "Unique HGNC + MONDO relationships:",
    len(gencc_relationship_coverage)
)

print(
    "Relationships with GenCC assertions:",
    gencc_relationship_coverage[
        "gencc_match"
    ].sum()
)

print(
    "Relationships without GenCC assertions:",
    (
        ~gencc_relationship_coverage[
            "gencc_match"
        ]
    ).sum()
)

# ============================================================
# Calculate combined inheritance-source coverage
# ============================================================

clingen_coverage = (
    relationship_match_summary[
        [
            "HGNC_ID",
            "MONDO_ID",
            "clingen_match",
        ]
    ]
    .drop_duplicates()
)

combined_coverage = (
    clingen_coverage
    .merge(
        gencc_relationship_coverage[
            [
                "HGNC_ID",
                "MONDO_ID",
                "gencc_match",
            ]
        ],
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="outer",
    )
)

combined_coverage[
    "clingen_match"
] = (
    combined_coverage[
        "clingen_match"
    ]
    .fillna(False)
)

combined_coverage[
    "gencc_match"
] = (
    combined_coverage[
        "gencc_match"
    ]
    .fillna(False)
)

gencc_added_coverage = (
    ~combined_coverage["clingen_match"]
    & combined_coverage["gencc_match"]
)

print(
    "\nRelationships added by GenCC beyond direct ClinGen:",
    gencc_added_coverage.sum()
)

print(
    "Relationships covered by either source:",
    (
        combined_coverage["clingen_match"]
        | combined_coverage["gencc_match"]
    ).sum()
)

# ============================================================
# Build unified inheritance evidence table
# ============================================================

inheritance_evidence = pd.concat(
    [
        clingen_evidence,
        gencc_evidence,
    ],
    ignore_index=True,
    sort=False,
)

inheritance_evidence = (
    inheritance_evidence
    .drop_duplicates(
        subset=[
            "HGNC_ID",
            "MONDO_ID",
            "inheritance_model",
            "gene_disease_validity",
            "inheritance_submitter",
        ]
    )
    .reset_index(drop=True)
)

# ============================================================
# Summarise inheritance evidence by disease and model
# ============================================================

inheritance_model_summary = (
    inheritance_evidence
    .groupby(
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "MONDO_ID",
            "inheritance_model",
            "inheritance_label",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        best_validity_rank=(
            "validity_rank",
            "max",
        ),
        evidence_sources=(
            "inheritance_submitter",
            combine_unique_values,
        ),
        validity_evidence=(
            "gene_disease_validity",
            combine_unique_values,
        ),
        assertion_count=(
            "gene_disease_validity",
            "size",
        ),
    )
)

inheritance_model_summary[
    "best_validity"
] = (
    inheritance_model_summary[
        "best_validity_rank"
    ]
    .map(RANK_TO_VALIDITY)
)

# Flag inheritance models for which evidence sources report
# more than one validity classification.
inheritance_model_summary[
    "validity_variation"
] = (
    inheritance_model_summary[
        "validity_evidence"
    ]
    .fillna("")
    .str.contains(
        r"\|",
        regex=True,
    )
)

print("\nUnified inheritance model coverage")
print("----------------------------------")

print(
    inheritance_model_summary[
        "inheritance_model"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print(
    "\nInheritance model rows with variation in validity evidence:",
    inheritance_model_summary[
        "validity_variation"
    ].sum()
)

# ============================================================
# Summarise inheritance-model agreement per gene-disease pair
# ============================================================

inheritance_relationship_summary = (
    inheritance_model_summary
    .groupby(
        [
            "HGNC_ID",
            "MONDO_ID",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        asserted_inheritance_models=(
            "inheritance_model",
            combine_unique_values,
        ),
        number_of_asserted_models=(
            "inheritance_model",
            "nunique",
        ),
    )
)

inheritance_relationship_summary[
    "multiple_inheritance_models"
] = (
    inheritance_relationship_summary[
        "number_of_asserted_models"
    ] > 1
)

inheritance_model_summary = (
    inheritance_model_summary
    .merge(
        inheritance_relationship_summary,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="left",
        validate="m:1",
    )
)

x_linked_relationships = (
    inheritance_model_summary[
        inheritance_model_summary[
            "inheritance_model"
        ].isin(X_LINKED_MODELS)
    ][
        [
            "HGNC_ID",
            "MONDO_ID",
        ]
    ]
    .drop_duplicates()
)

x_test_candidates = (
    pvars[
        pvars["Chromosome"].astype(str) == "X"
    ]
    .merge(
        x_linked_relationships,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="inner",
    )
)

# ============================================================
# Retrieve variant frequencies from gnomAD
# ============================================================

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"

GNOMAD_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://gnomad.broadinstitute.org",
    "Referer": "https://gnomad.broadinstitute.org/",
}

GNOMAD_QUERY = """
query VariantPopulationFrequencies(
    $variantId: String!,
    $dataset: DatasetId!
) {
    variant(
        variantId: $variantId,
        dataset: $dataset
    ) {
        variant_id

        exome {
            ac
            an
            af

            populations {
                id
                ac
                an
            }
        }

        genome {
            ac
            an
            af

            populations {
                id
                ac
                an
            }
        }
    }
}
"""

RETRYABLE_STATUS_CODES = {
    403,
    429,
    500,
    502,
    503,
    504,
}

def initialise_gnomad_cache(cache_file=GNOMAD_CACHE_FILE):
    """
    Create the SQLite gnomAD cache and required tables.

    The cache contains:
      1. One query-status row per variant and dataset.
      2. Raw population frequency rows from exomes and genomes.
    """

    with sqlite3.connect(cache_file) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_status (
                variant_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                queried_at TEXT NOT NULL,

                PRIMARY KEY (variant_id, dataset)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS population_frequencies (
                variant_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                source TEXT NOT NULL,
                population_id TEXT NOT NULL,
                ac INTEGER,
                an INTEGER,
                af REAL,

                PRIMARY KEY (
                    variant_id,
                    dataset,
                    source,
                    population_id
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frequency_variant
            ON population_frequencies (
                variant_id,
                dataset
            )
            """
        )

        connection.commit()

def get_cached_status(
    variant_id,
    dataset=GNOMAD_DATASET,
    cache_file=GNOMAD_CACHE_FILE,
):
    """
    Return the cached status for a variant, or None if the variant
    has not previously completed successfully.
    """

    with sqlite3.connect(cache_file) as connection:

        result = connection.execute(
            """
            SELECT status, error, queried_at
            FROM query_status
            WHERE variant_id = ?
              AND dataset = ?
            """,
            (
                variant_id,
                dataset,
            ),
        ).fetchone()

    if result is None:
        return None

    return {
        "status": result[0],
        "error": result[1],
        "queried_at": result[2],
    }

def load_cached_query_status(
    variant_ids=None,
    dataset=GNOMAD_DATASET,
    cache_file=GNOMAD_CACHE_FILE,
):
    """
    Load cached gnomAD query status information.
    """

    with sqlite3.connect(cache_file) as connection:

        status_df = pd.read_sql_query(
            """
            SELECT
                variant_id,
                status AS gnomad_query_status,
                error AS gnomad_query_error,
                queried_at
            FROM query_status
            WHERE dataset = ?
            """,
            connection,
            params=(dataset,),
        )

    if variant_ids is not None:
        variant_ids = set(variant_ids)

        status_df = status_df[
            status_df["variant_id"].isin(
                variant_ids
            )
        ].copy()

    return status_df

def save_gnomad_result_to_cache(
    query_result,
    frequency_df,
    dataset=GNOMAD_DATASET,
    cache_file=GNOMAD_CACHE_FILE,
):
    """
    Save one completed gnomAD query to SQLite.

    Only terminal data outcomes are cached:
        success
        not_found

    Request and GraphQL failures are not cached, allowing them
    to be retried on the next run.
    """

    status = query_result["status"]

    if status not in {"success", "not_found"}:
        return False

    variant_id = query_result["variant_id"]

    queried_at = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(cache_file) as connection:

        try:
            connection.execute("BEGIN")

            # Remove old population rows before replacing a
            # successfully re-queried variant.
            if status == "success":
                connection.execute(
                    """
                    DELETE FROM population_frequencies
                    WHERE variant_id = ?
                      AND dataset = ?
                    """,
                    (
                        variant_id,
                        dataset,
                    ),
                )

                if not frequency_df.empty:

                    rows_to_insert = []

                    for row in frequency_df.itertuples(index=False):

                        ac = (
                            None
                            if pd.isna(row.ac)
                            else int(row.ac)
                        )

                        an = (
                            None
                            if pd.isna(row.an)
                            else int(row.an)
                        )

                        af = (
                            None
                            if pd.isna(row.af)
                            else float(row.af)
                        )

                        rows_to_insert.append(
                            (
                                variant_id,
                                dataset,
                                str(row.source),
                                str(row.population_id),
                                ac,
                                an,
                                af,
                            )
                        )

                    connection.executemany(
                        """
                        INSERT OR REPLACE INTO population_frequencies (
                            variant_id,
                            dataset,
                            source,
                            population_id,
                            ac,
                            an,
                            af
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows_to_insert,
                    )

            connection.execute(
                """
                INSERT OR REPLACE INTO query_status (
                    variant_id,
                    dataset,
                    status,
                    error,
                    queried_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    variant_id,
                    dataset,
                    status,
                    query_result.get("error"),
                    queried_at,
                ),
            )

            connection.commit()
            return True

        except Exception:
            connection.rollback()
            raise

def load_cached_frequencies(
    variant_ids=None,
    dataset=GNOMAD_DATASET,
    cache_file=GNOMAD_CACHE_FILE,
):
    """
    Load raw gnomAD population frequencies from SQLite.

    If variant_ids is None, load every cached frequency row.
    """

    with sqlite3.connect(cache_file) as connection:

        if variant_ids is None:

            frequency_df = pd.read_sql_query(
                """
                SELECT
                    variant_id,
                    source,
                    population_id,
                    ac,
                    an,
                    af
                FROM population_frequencies
                WHERE dataset = ?
                """,
                connection,
                params=(dataset,),
            )

        else:

            variant_ids = list(dict.fromkeys(variant_ids))

            if not variant_ids:
                return pd.DataFrame(
                    columns=[
                        "variant_id",
                        "source",
                        "population_id",
                        "ac",
                        "an",
                        "af",
                    ]
                )

            placeholders = ",".join(
                ["?"] * len(variant_ids)
            )

            query = f"""
                SELECT
                    variant_id,
                    source,
                    population_id,
                    ac,
                    an,
                    af
                FROM population_frequencies
                WHERE dataset = ?
                  AND variant_id IN ({placeholders})
            """

            frequency_df = pd.read_sql_query(
                query,
                connection,
                params=[
                    dataset,
                    *variant_ids,
                ],
            )

    return frequency_df

def query_gnomad_variant(
    variant_id,
    session,
    dataset=GNOMAD_DATASET,
    max_retries=5,
    timeout=60,
    base_sleep=1.0,
    jitter=0.5,
):

    payload = {
        "query": GNOMAD_QUERY,
        "variables": {
            "variantId": variant_id,
            "dataset": dataset,
        },
    }

    for attempt in range(1, max_retries + 1):

        try:

            response = session.post(
                GNOMAD_API_URL,
                json=payload,
                headers=GNOMAD_HEADERS,
                timeout=timeout,
            )

            # ------------------------------------------
            # Successful HTTP response
            # ------------------------------------------

            if response.status_code == 200:

                result = response.json()

                # GraphQL can return HTTP 200
                # while containing GraphQL errors.
                if result.get("errors"):

                    error_messages = [
                        error.get("message", str(error))
                        for error in result["errors"]
                    ]

                    error_text = " | ".join(error_messages)

                    # A valid variant query may simply not be present in gnomAD.
                    # Treat this as a data outcome rather than an API failure.
                    if any(
                        "Variant not found" in message
                        for message in error_messages
                    ):
                        return {
                            "variant_id": variant_id,
                            "status": "not_found",
                            "data": None,
                            "error": None,
                        }

                    return {
                        "variant_id": variant_id,
                        "status": "graphql_error",
                        "data": None,
                        "error": error_text,
                    }

                variant_data = (
                    result
                    .get("data", {})
                    .get("variant")
                )

                if variant_data is None:

                    return {
                        "variant_id": variant_id,
                        "status": "not_found",
                        "data": None,
                        "error": None,
                    }

                return {
                    "variant_id": variant_id,
                    "status": "success",
                    "data": variant_data,
                    "error": None,
                }

            # ------------------------------------------
            # Cloudflare / rate limiting / server error
            # ------------------------------------------

            if response.status_code in RETRYABLE_STATUS_CODES:

                if attempt == max_retries:

                    return {
                        "variant_id": variant_id,
                        "status": "request_failed",
                        "data": None,
                        "error": (
                            f"HTTP {response.status_code}: "
                            f"{response.text[:500]}"
                        ),
                    }

                sleep_time = (
                    base_sleep * (2 ** (attempt - 1))
                    + random.uniform(0, jitter)
                )

                print(
                    f"[WARN] {variant_id}: "
                    f"gnomAD returned {response.status_code}. "
                    f"Retrying in {sleep_time:.2f}s "
                    f"({attempt}/{max_retries})"
                )

                time.sleep(sleep_time)
                continue

            # ------------------------------------------
            # Non-retryable HTTP error
            # ------------------------------------------

            return {
                "variant_id": variant_id,
                "status": "request_failed",
                "data": None,
                "error": (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                ),
            }

        except requests.exceptions.RequestException as exc:

            if attempt == max_retries:

                return {
                    "variant_id": variant_id,
                    "status": "request_failed",
                    "data": None,
                    "error": str(exc),
                }

            sleep_time = (
                base_sleep * (2 ** (attempt - 1))
                + random.uniform(0, jitter)
            )

            print(
                f"[WARN] {variant_id}: network error. "
                f"Retrying in {sleep_time:.2f}s."
            )

            time.sleep(sleep_time)

    return {
        "variant_id": variant_id,
        "status": "request_failed",
        "data": None,
        "error": "Maximum retries exceeded",
    }

def normalise_gnomad_result(query_result):
    """
    Convert one successful gnomAD result into long-format rows.

    One row represents one:
        variant + dataset source + population
    """

    variant_id = query_result["variant_id"]
    status = query_result["status"]

    if status != "success":
        return pd.DataFrame(
            columns=[
                "variant_id",
                "source",
                "population_id",
                "ac",
                "an",
                "af",
            ]
        )

    variant_data = query_result["data"]
    rows = []

    for source in ["exome", "genome"]:
        source_data = variant_data.get(source)

        if source_data is None:
            continue

        populations = source_data.get("populations") or []

        for population in populations:
            population_id = population.get("id")
            ac = population.get("ac")
            an = population.get("an")

            if population_id is None:
                continue

            if ac is not None and an not in (None, 0):
                af = ac / an
            else:
                af = pd.NA

            rows.append(
                {
                    "variant_id": variant_id,
                    "source": source,
                    "population_id": population_id,
                    "ac": ac,
                    "an": an,
                    "af": af,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "variant_id",
            "source",
            "population_id",
            "ac",
            "an",
            "af",
        ],
    )

def select_gnomad_frequencies(
    frequency_df,
    min_population_an=MIN_POPULATION_AN,
):
    """
    Select one usable gnomAD frequency estimate for each
    variant/population combination.

    Selection rules
    ---------------
    1. Population AN must be at least min_population_an.
    2. Exclude population groups without a useful population label.
    3. Prefer exome where sufficient exome data exist.
    4. Fall back to genome where exome AN is insufficient.
    5. Exclude the population if neither source has sufficient AN.

    The original raw frequency table is not modified.
    """

    if frequency_df.empty:
        return frequency_df.copy()

    df = frequency_df.copy()

    # Ensure numeric comparisons work after loading from SQLite.
    df["an"] = pd.to_numeric(df["an"], errors="coerce")
    df["ac"] = pd.to_numeric(df["ac"], errors="coerce")
    df["af"] = pd.to_numeric(df["af"], errors="coerce")

    # Require sufficient allele number.
    df = df[
        df["an"].notna()
        & (df["an"] >= min_population_an)
    ].copy()

    # Exclude labels that are not interpretable populations.
    df = df[
        ~df["population_id"].isin(EXCLUDED_POPULATION_IDS)
    ].copy()

    if df.empty:
        return df.reset_index(drop=True)

    # Prefer exome, falling back to genome.
    source_priority = {
        "exome": 1,
        "genome": 2,
    }

    df["source_priority"] = (
        df["source"]
        .map(source_priority)
        .fillna(99)
    )

    df = df.sort_values(
        by=[
            "variant_id",
            "population_id",
            "source_priority",
        ]
    )

    selected = df.drop_duplicates(
        subset=[
            "variant_id",
            "population_id",
        ],
        keep="first",
    ).copy()

    selected = selected.drop(
        columns=["source_priority"]
    )

    return selected.reset_index(drop=True)

# ============================================================
# Query gnomAD with persistent SQLite caching
# ============================================================

initialise_gnomad_cache()

all_variants = (
    pvars["variant"]
    .dropna()
    .astype("string")
    .drop_duplicates()
    .tolist()
)

if GNOMAD_TEST_LIMIT is None:
    variants_to_process = all_variants
else:
    variants_to_process = all_variants[:GNOMAD_TEST_LIMIT]





# During development, ensure that a few curated X-linked
# variants are represented in the test set.
if GNOMAD_TEST_LIMIT is not None:

    x_test_variants = (
        x_test_candidates[
            "variant"
        ]
        .dropna()
        .astype("string")
        .drop_duplicates()
        .head(15)
        .tolist()
    )

    variants_to_process = list(
        dict.fromkeys(
            variants_to_process
            + x_test_variants
        )
    )

# Check which X-linked development candidates are already cached.
cached_x_status = load_cached_query_status(
    variant_ids=(
        x_test_candidates[
            "variant"
        ]
        .dropna()
        .astype("string")
        .drop_duplicates()
        .tolist()
    )
)



print(
    f"Processing {len(variants_to_process):,} "
    "gnomAD variants."
)

query_log = []

cached_count = 0
queried_count = 0
success_count = 0
not_found_count = 0
failure_count = 0

with requests.Session() as session:
    for number, variant_id in enumerate(
        variants_to_process,
        start=1,
    ):
        cached_status = get_cached_status(
            variant_id=variant_id
        )

        # Reuse completed cached outcomes.
        if (
            not FORCE_GNOMAD_REFRESH
            and cached_status is not None
            and cached_status["status"] in {
                "success",
                "not_found",
            }
            and cache_is_current(cached_status)
        ):
            cached_count += 1

            query_log.append(
                {
                    "variant_id": variant_id,
                    "status": cached_status["status"],
                    "result_source": "cache",
                    "error": cached_status["error"],
                }
            )

            continue

        if FORCE_GNOMAD_REFRESH:
            query_reason = "forced refresh"

        elif cached_status is None:
            query_reason = "not cached"

        elif not cache_is_current(cached_status):
            query_reason = "cache expired"

        else:
            query_reason = "retrying previous failure"

        # Query variants not already cached.
        print(
            f"[{number}/{len(variants_to_process)}] "
            f"Querying: {variant_id} "
            f"({query_reason})"
        )

        query_result = query_gnomad_variant(
            variant_id=variant_id,
            session=session,
        )

        queried_count += 1

        raw_frequency_df = normalise_gnomad_result(
            query_result
        )

        was_cached = save_gnomad_result_to_cache(
            query_result=query_result,
            frequency_df=raw_frequency_df,
        )

        status = query_result["status"]

        if status == "success":
            success_count += 1
        elif status == "not_found":
            not_found_count += 1
        else:
            failure_count += 1

        query_log.append(
            {
                "variant_id": variant_id,
                "status": status,
                "result_source": (
                    "api_saved"
                    if was_cached
                    else "api_not_cached"
                ),
                "error": query_result["error"],
            }
        )

        # Be considerate of the public gnomAD API.
        time.sleep(
            1.0 + random.uniform(0, 0.5)
        )


# ============================================================
# Load cached results and select usable frequencies
# ============================================================

gnomad_frequency_raw = load_cached_frequencies(
    variant_ids=variants_to_process
)

gnomad_frequency_selected = select_gnomad_frequencies(
    gnomad_frequency_raw
)

gnomad_query_log = pd.DataFrame(query_log)

gnomad_cached_status = load_cached_query_status(
    variant_ids=variants_to_process
)

# Include non-cached failures from the current run so that they
# remain distinguishable from variants that were not evaluated.
current_run_status = gnomad_query_log[
    [
        "variant_id",
        "status",
        "error",
    ]
].rename(
    columns={
        "status": "gnomad_query_status",
        "error": "gnomad_query_error",
    }
)

current_run_status["queried_at"] = pd.NA

gnomad_cached_status = pd.concat(
    [
        gnomad_cached_status,
        current_run_status,
    ],
    ignore_index=True,
)

# Prefer the SQLite record when a completed cached result and
# a current-run log entry both exist.
gnomad_cached_status["status_priority"] = np.where(
    gnomad_cached_status["queried_at"].notna(),
    1,
    2,
)

gnomad_cached_status = (
    gnomad_cached_status
    .sort_values(
        [
            "variant_id",
            "status_priority",
        ]
    )
    .drop_duplicates(
        subset=["variant_id"],
        keep="first",
    )
    .drop(
        columns=["status_priority"]
    )
    .reset_index(drop=True)
)

# ============================================================
# Development summary
# ============================================================

print("\nRun summary")
print("-----------")
print(
    f"Variants requested: "
    f"{len(variants_to_process):,}"
)
print(
    f"Reused from cache: "
    f"{cached_count:,}"
)
print(
    f"Queried from API: "
    f"{queried_count:,}"
)
print(
    f"New successful results: "
    f"{success_count:,}"
)
print(
    f"New not-found results: "
    f"{not_found_count:,}"
)
print(
    f"Uncached failures: "
    f"{failure_count:,}"
)

print("\nSelected frequency sources:")
if gnomad_frequency_selected.empty:
    print("No usable population frequencies.")
else:
    print(
        gnomad_frequency_selected["source"]
        .value_counts()
        .to_string()
    )

print(
    f"\nMinimum population AN: "
    f"{MIN_POPULATION_AN:,}"
)

print(
    "Raw cached population rows:",
    len(gnomad_frequency_raw),
)

print(
    "Selected population rows:",
    len(gnomad_frequency_selected),
)

# ============================================================
# Join ClinVar annotations to selected gnomAD frequencies
# ============================================================

clinvar_gnomad = pvars.merge(
    gnomad_frequency_selected,
    left_on="variant",
    right_on="variant_id",
    how="inner",
    validate="m:m",
)

print("\nClinVar + gnomAD joined data")
print("----------------------------")

print(
    f"Rows: {len(clinvar_gnomad):,}"
)

print(
    "Unique variants:",
    clinvar_gnomad["variant"].nunique()
)

print(
    "Unique genes:",
    clinvar_gnomad["GeneSymbol"].nunique()
)

print(
    "Unique phenotypes:",
    clinvar_gnomad["PhenotypeList"].nunique()
)

frequency_analysis = clinvar_gnomad[
    [
        "variant",
        "GeneSymbol",
        "HGNC_ID",
        "PhenotypeList",
        "Disease_ID",
        "MONDO_ID",
        "Chromosome",
        "PositionVCF",
        "population_id",
        "source",
        "ac",
        "an",
        "af",
    ]
].copy()

frequency_analysis = frequency_analysis.drop_duplicates()

print(
    "\nAnalysis rows after removing exact duplicates:",
    len(frequency_analysis)
)

def split_population_and_sex(population_id):
    """
    Separate a gnomAD population ID into population and sex.

    Examples
    --------
    nfe     -> nfe, all
    nfe_XX  -> nfe, XX
    nfe_XY  -> nfe, XY
    XX      -> overall, XX
    XY      -> overall, XY
    """

    if pd.isna(population_id):
        return pd.NA, pd.NA

    population_id = str(population_id)

    if population_id == "XX":
        return "overall", "XX"

    if population_id == "XY":
        return "overall", "XY"

    if population_id.endswith("_XX"):
        return population_id[:-3], "XX"

    if population_id.endswith("_XY"):
        return population_id[:-3], "XY"

    return population_id, "all"

population_sex = (
    frequency_analysis["population_id"]
    .apply(split_population_and_sex)
)

frequency_analysis[
    ["population", "sex"]
] = pd.DataFrame(
    population_sex.tolist(),
    index=frequency_analysis.index,
)

print("\nPopulation/sex combinations")
print("---------------------------")

population_sex_check = (
    frequency_analysis[
        [
            "population_id",
            "population",
            "sex",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        by=[
            "population",
            "sex",
        ]
    )
)

print(
    population_sex_check.to_string(
        index=False
    )
)

duplicate_mask = frequency_analysis.duplicated(
    subset=[
        "variant",
        "HGNC_ID",
        "Disease_ID",
        "PhenotypeList",
        "population",
        "sex",
    ],
    keep=False,
)

duplicate_rows = frequency_analysis[
    duplicate_mask
].copy()

print(
    "\nDuplicate "
    "variant/gene/phenotype/population/sex rows:",
    len(duplicate_rows)
)

# ============================================================
# Build ClinVar variant evidence table
# ============================================================

# Keep every unique ClinVar variant/gene/phenotype relationship.
clinvar_evidence_base = (
    pvars[
        [
            "variant",
            "GeneSymbol",
            "PhenotypeList",
            "Chromosome",
            "ClinicalSignificance",
        ]
    ]
    .drop_duplicates()
    .copy()
)

# During development, include only variants that were part of
# the current limited gnomAD run.
clinvar_evidence_base = clinvar_evidence_base[
    clinvar_evidence_base["variant"].isin(
        variants_to_process
    )
].copy()

print(
    "\nClinVar evidence relationships in current run:",
    len(clinvar_evidence_base)
)


# ------------------------------------------------------------
# Add cached gnomAD query status
# ------------------------------------------------------------

variant_evidence = clinvar_evidence_base.merge(
    gnomad_cached_status,
    left_on="variant",
    right_on="variant_id",
    how="left",
    validate="m:1",
)

variant_evidence = variant_evidence.drop(
    columns=["variant_id"]
)


# ------------------------------------------------------------
# Summarise usable gnomAD frequency information by variant
# ------------------------------------------------------------

variant_frequency_summary = (
    gnomad_frequency_selected
    .groupby(
        "variant_id",
        as_index=False,
    )
    .agg(
        maximum_observed_af=(
            "af",
            "max",
        ),
        populations_with_usable_data=(
            "population_id",
            "nunique",
        ),
    )
)


# ------------------------------------------------------------
# Add usable frequency summary to the evidence table
# ------------------------------------------------------------

variant_evidence = variant_evidence.merge(
    variant_frequency_summary,
    left_on="variant",
    right_on="variant_id",
    how="left",
    validate="m:1",
)

variant_evidence = variant_evidence.drop(
    columns=["variant_id"]
)


# ------------------------------------------------------------
# Initialise evidence classification
# ------------------------------------------------------------

variant_evidence["evidence_status"] = pd.NA
variant_evidence["evidence_reason"] = pd.NA


# ------------------------------------------------------------
# 1. Variant was not found in gnomAD
# ------------------------------------------------------------

not_found_mask = (
    variant_evidence["gnomad_query_status"]
    == "not_found"
)

variant_evidence.loc[
    not_found_mask,
    "evidence_status",
] = STATUS_GNOMAD_NOT_FOUND

variant_evidence.loc[
    not_found_mask,
    "evidence_reason",
] = "Variant not found in gnomAD"


# ------------------------------------------------------------
# 2. Variant query failed
# ------------------------------------------------------------

failure_mask = (
    variant_evidence["gnomad_query_status"].notna()
    & ~variant_evidence[
        "gnomad_query_status"
    ].isin(
        [
            "success",
            "not_found",
        ]
    )
)

variant_evidence.loc[
    failure_mask,
    "evidence_status",
] = STATUS_QUERY_FAILED

variant_evidence.loc[
    failure_mask,
    "evidence_reason",
] = (
    "gnomAD query did not complete successfully"
)


# ------------------------------------------------------------
# 3. Variant exists, but no usable population passes AN
# ------------------------------------------------------------

insufficient_an_mask = (
    (
        variant_evidence["gnomad_query_status"]
        == "success"
    )
    & variant_evidence[
        "maximum_observed_af"
    ].isna()
)

variant_evidence.loc[
    insufficient_an_mask,
    "evidence_status",
] = STATUS_INSUFFICIENT_AN

variant_evidence.loc[
    insufficient_an_mask,
    "evidence_reason",
] = (
    f"No population estimate with "
    f"AN >= {MIN_POPULATION_AN:,}"
)


# ------------------------------------------------------------
# 4. Variant is too common for the penetrant model
# ------------------------------------------------------------

common_mask = (
    variant_evidence[
        "maximum_observed_af"
    ].notna()
    & (
        variant_evidence[
            "maximum_observed_af"
        ]
        >= MAX_MODEL_ELIGIBLE_AF
    )
)

variant_evidence.loc[
    common_mask,
    "evidence_status",
] = STATUS_EXCLUDED_COMMON

variant_evidence.loc[
    common_mask,
    "evidence_reason",
] = (
    "Variant exceeds rare, highly penetrant "
    "model AF threshold in at least one "
    "well-powered population"
)


# ------------------------------------------------------------
# 5. Remaining variants with usable frequencies are included
# ------------------------------------------------------------

included_mask = (
    (
        variant_evidence["gnomad_query_status"]
        == "success"
    )
    & variant_evidence[
        "maximum_observed_af"
    ].notna()
    & variant_evidence[
        "evidence_status"
    ].isna()
)

variant_evidence.loc[
    included_mask,
    "evidence_status",
] = STATUS_INCLUDED

variant_evidence.loc[
    included_mask,
    "evidence_reason",
] = (
    "Included in rare, highly penetrant allele model"
)


# ------------------------------------------------------------
# 6. Catch anything that could not be classified
# ------------------------------------------------------------

unclassified_mask = (
    variant_evidence["evidence_status"].isna()
)

variant_evidence.loc[
    unclassified_mask,
    "evidence_status",
] = "not_evaluated"

variant_evidence.loc[
    unclassified_mask,
    "evidence_reason",
] = (
    "Variant has not yet been fully evaluated"
)


# ============================================================
# Create Dash-ready variant evidence tables
# ============================================================

# ClinVar Pathogenic variants used in the prevalence estimate.
evidence_included = variant_evidence[
    variant_evidence["evidence_status"]
    == STATUS_INCLUDED
].copy()


# ClinVar Pathogenic variants without usable gnomAD frequency.
evidence_without_frequency = variant_evidence[
    variant_evidence[
        "evidence_status"
    ].isin(
        [
            STATUS_GNOMAD_NOT_FOUND,
            STATUS_INSUFFICIENT_AN,
        ]
    )
].copy()


# ClinVar Pathogenic variants excluded because their maximum
# reliable population AF exceeds the model threshold.
evidence_excluded_common = variant_evidence[
    variant_evidence["evidence_status"]
    == STATUS_EXCLUDED_COMMON
].copy()


# Operational failures are tracked separately from biological
# absence or insufficient population data.
evidence_query_failed = variant_evidence[
    variant_evidence["evidence_status"]
    == STATUS_QUERY_FAILED
].copy()


# Variants that were not fully classified.
evidence_not_evaluated = variant_evidence[
    variant_evidence["evidence_status"]
    == "not_evaluated"
].copy()


# ============================================================
# Evidence QC summary
# ============================================================

print("\nVariant evidence summary")
print("------------------------")

# Count unique variants rather than ClinVar phenotype rows.
evidence_summary = (
    variant_evidence[
        [
            "variant",
            "evidence_status",
        ]
    ]
    .drop_duplicates()
    ["evidence_status"]
    .value_counts()
)

print(
    evidence_summary.to_string()
)

print(
    "\nUnique variants included:",
    evidence_included["variant"].nunique()
)

print(
    "Unique variants without usable gnomAD frequency:",
    evidence_without_frequency["variant"].nunique()
)

print(
    "Unique variants excluded as common:",
    evidence_excluded_common["variant"].nunique()
)

print(
    "Unique variants with query failures:",
    evidence_query_failed["variant"].nunique()
)

print(
    "Unique variants not evaluated:",
    evidence_not_evaluated["variant"].nunique()
)


# ------------------------------------------------------------
# Reconciliation check
# ------------------------------------------------------------

processed_unique_variants = len(
    set(variants_to_process)
)

classified_unique_variants = (
    variant_evidence[
        [
            "variant",
            "evidence_status",
        ]
    ]
    .drop_duplicates()
    ["variant"]
    .nunique()
)

print(
    "\nUnique processed variants:",
    processed_unique_variants
)

print(
    "Unique variants represented in evidence table:",
    classified_unique_variants
)

# ============================================================
# Calculate autosomal pathogenic allele burden
# ============================================================

autosomal_frequency_analysis = frequency_analysis[
    (frequency_analysis["Chromosome"].astype(str) != "X")
    & (frequency_analysis["sex"] == "all")
].copy()

print(
    "\nAutosomal variant/population observations:",
    len(autosomal_frequency_analysis)
)

# ============================================================
# Apply variant-level evidence classification
# ============================================================

included_variant_ids = set(
    evidence_included["variant"].unique()
)

excluded_common_variant_ids = set(
    evidence_excluded_common["variant"].unique()
)


# Population-frequency observations allowed to contribute
# to the prevalence calculation.
autosomal_model_data = autosomal_frequency_analysis[
    autosomal_frequency_analysis["variant"].isin(
        included_variant_ids
    )
].copy()


# Population-frequency observations excluded because the
# variant exceeds the AF threshold in at least one
# sufficiently powered population.
autosomal_flagged_common = autosomal_frequency_analysis[
    autosomal_frequency_analysis["variant"].isin(
        excluded_common_variant_ids
    )
].copy()

# ============================================================
# Report common variants excluded from the autosomal model
# ============================================================

autosomal_flagged_common[
    "exclusion_reason"
] = (
    "Variant exceeds rare, highly penetrant "
    "model AF threshold in at least one "
    "well-powered population"
)

print(
    "\nAutosomal observations excluded from the "
    "rare, highly penetrant model:",
    len(autosomal_flagged_common)
)

print(
    "Unique variants excluded from the autosomal model:",
    autosomal_flagged_common["variant"].nunique()
)

if not autosomal_flagged_common.empty:

    print(
        autosomal_flagged_common[
            [
                "variant",
                "GeneSymbol",
                "PhenotypeList",
                "population",
                "source",
                "ac",
                "an",
                "af",
                "exclusion_reason",
            ]
        ]
        .sort_values(
            "af",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

# ============================================================
# Calculate eligible autosomal pathogenic allele burden
# ============================================================

autosomal_allele_burden = (
    autosomal_model_data
    .groupby(
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "Chromosome",
            "population",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        pathogenic_allele_frequency=(
            "af",
            "sum",
        ),
        number_of_variants=(
            "variant",
            "nunique",
        ),
        minimum_an=(
            "an",
            "min",
        ),
    )
)

autosomal_allele_burden["invalid_q"] = (
    autosomal_allele_burden[
        "pathogenic_allele_frequency"
    ] > 1
)

print("\nAutosomal allele burden summary")
print("-------------------------------")

print(
    "Gene/phenotype/population combinations:",
    len(autosomal_allele_burden)
)

print(
    "Combinations with q > 1:",
    autosomal_allele_burden[
        "invalid_q"
    ].sum()
)



# ============================================================
# Summarise variants excluded from the model
# ============================================================

flagged_common_summary = (
    autosomal_flagged_common
    .groupby(
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "population",
        ],
        as_index=False,
    )
    .agg(
        flagged_common_variants=(
            "variant",
            "nunique",
        ),
        maximum_flagged_af=(
            "af",
            "max",
        ),
    )
)

autosomal_allele_burden = (
    autosomal_allele_burden
    .merge(
        flagged_common_summary,
        on=[
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "population",
        ],
        how="outer",
    )
)

autosomal_allele_burden[
    "flagged_common_variants"
] = (
    autosomal_allele_burden[
        "flagged_common_variants"
    ]
    .fillna(0)
    .astype(int)
)

print("\nLargest autosomal pathogenic allele burdens:")

print(
    autosomal_allele_burden
    .sort_values(
        "pathogenic_allele_frequency",
        ascending=False,
    )
    .head(30)
    .to_string(index=False)
)

# ============================================================
# Calculate autosomal Hardy-Weinberg estimates
# ============================================================

q = autosomal_allele_burden[
    "pathogenic_allele_frequency"
]

# Frequency of individuals carrying exactly one
# pathogenic allele.
autosomal_allele_burden[
    "heterozygous_carrier_frequency"
] = (
    2 * q * (1 - q)
)

# Frequency of individuals expected to carry two
# pathogenic alleles under Hardy-Weinberg equilibrium.
autosomal_allele_burden[
    "biallelic_genotype_frequency"
] = q ** 2

# Frequency of individuals carrying at least one
# pathogenic allele.
autosomal_allele_burden[
    "any_pathogenic_genotype_frequency"
] = (
    2 * q * (1 - q)
    + q ** 2
)

PER_100K = 100_000

autosomal_allele_burden[
    "heterozygous_carriers_per_100k"
] = (
    autosomal_allele_burden[
        "heterozygous_carrier_frequency"
    ]
    * PER_100K
)

autosomal_allele_burden[
    "biallelic_genotypes_per_100k"
] = (
    autosomal_allele_burden[
        "biallelic_genotype_frequency"
    ]
    * PER_100K
)

autosomal_allele_burden[
    "any_pathogenic_genotype_per_100k"
] = (
    autosomal_allele_burden[
        "any_pathogenic_genotype_frequency"
    ]
    * PER_100K
)

# ============================================================
# Build inheritance-aware autosomal prevalence results
# ============================================================

autosomal_prevalence_base = (
    autosomal_allele_burden[
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "Chromosome",
            "population",
            "pathogenic_allele_frequency",
            "number_of_variants",
            "minimum_an",
            "heterozygous_carriers_per_100k",
            "biallelic_genotypes_per_100k",
            "any_pathogenic_genotype_per_100k",
        ]
    ]
    .copy()
)

autosomal_recessive_results = (
    autosomal_prevalence_base.copy()
)

autosomal_recessive_results[
    "inheritance_model"
] = "autosomal_recessive"

autosomal_recessive_results[
    "inheritance_label"
] = "Autosomal recessive"

autosomal_recessive_results[
    "estimated_genetic_prevalence_per_100k"
] = (
    autosomal_recessive_results[
        "biallelic_genotypes_per_100k"
    ]
)

autosomal_dominant_results = (
    autosomal_prevalence_base.copy()
)

autosomal_dominant_results[
    "inheritance_model"
] = "autosomal_dominant"

autosomal_dominant_results[
    "inheritance_label"
] = "Autosomal dominant"

autosomal_dominant_results[
    "estimated_genetic_prevalence_per_100k"
] = (
    autosomal_dominant_results[
        "any_pathogenic_genotype_per_100k"
    ]
)

autosomal_prevalence_results = pd.concat(
    [
        autosomal_recessive_results,
        autosomal_dominant_results,
    ],
    ignore_index=True,
)

# ============================================================
# Prepare model-specific inheritance evidence
# ============================================================

autosomal_inheritance_evidence = (
    inheritance_model_summary[
        inheritance_model_summary[
            "inheritance_model"
        ].isin(
            [
                "autosomal_recessive",
                "autosomal_dominant",
            ]
        )
    ][
        [
            "HGNC_ID",
            "MONDO_ID",
            "inheritance_model",
            "best_validity",
            "best_validity_rank",
            "validity_evidence",
            "evidence_sources",
            "assertion_count",
            "validity_variation",
        ]
    ]
    .drop_duplicates()
    .copy()
)

# Add evidence specific to the calculated inheritance model.
autosomal_prevalence_results = (
    autosomal_prevalence_results
    .merge(
        autosomal_inheritance_evidence,
        on=[
            "HGNC_ID",
            "MONDO_ID",
            "inheritance_model",
        ],
        how="left",
        validate="m:1",
    )
)

autosomal_prevalence_results = (
    autosomal_prevalence_results
    .merge(
        inheritance_relationship_summary,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="left",
        validate="m:1",
    )
)

no_inheritance_evidence = (
    autosomal_prevalence_results[
        "best_validity"
    ].isna()
)

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "best_validity",
] = "No supporting assertion"

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "best_validity_rank",
] = VALIDITY_RANK[
    "No supporting assertion"
]

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "validity_evidence",
] = "No supporting assertion"

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "assertion_count",
] = 0

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "evidence_sources",
] = "No supporting source"

autosomal_prevalence_results.loc[
    no_inheritance_evidence,
    "validity_variation",
] = False

autosomal_prevalence_results[
    "display_priority"
] = (
    autosomal_prevalence_results[
        "best_validity_rank"
    ]
)

no_relationship_inheritance = (
    autosomal_prevalence_results[
        "asserted_inheritance_models"
    ].isna()
)

autosomal_prevalence_results.loc[
    no_relationship_inheritance,
    "asserted_inheritance_models",
] = "none"

autosomal_prevalence_results.loc[
    no_relationship_inheritance,
    "number_of_asserted_models",
] = 0

autosomal_prevalence_results.loc[
    no_relationship_inheritance,
    "multiple_inheritance_models",
] = False

print("\nHOGA1 inheritance-aware prevalence example")
print("------------------------------------------")

hoga1_example = autosomal_prevalence_results[
    (
        autosomal_prevalence_results[
            "GeneSymbol"
        ] == "HOGA1"
    )
    & (
        autosomal_prevalence_results[
            "MONDO_ID"
        ] == "MONDO:0013327"
    )
    & (
        autosomal_prevalence_results[
            "population"
        ] == "asj"
    )
].copy()

hoga1_example = hoga1_example.sort_values(
    "display_priority",
    ascending=False,
)

print(
    hoga1_example[
        [
            "GeneSymbol",
            "PhenotypeList",
            "population",
            "inheritance_label",
            "estimated_genetic_prevalence_per_100k",
            "best_validity",
            "validity_evidence",
            "evidence_sources",
            "assertion_count",
            "validity_variation",
            "asserted_inheritance_models",
            "number_of_asserted_models",
            "multiple_inheritance_models",
        ]
    ].to_string(index=False)
)

# ============================================================
# Calculate X-chromosome pathogenic allele burden
# ============================================================

x_frequency_analysis = frequency_analysis[
    frequency_analysis["Chromosome"].astype(str) == "X"
].copy()

x_frequency_analysis[
    "is_par"
] = (
    x_frequency_analysis[
        "PositionVCF"
    ]
    .apply(is_grch38_par)
)

print(
    "X observations in PAR:",
    x_frequency_analysis[
        "is_par"
    ].fillna(False).sum()
)

print(
    "X observations outside PAR:",
    (
        ~x_frequency_analysis[
            "is_par"
        ].fillna(True)
    ).sum()
)

print(
    "\nX-chromosome variant/population observations:",
    len(x_frequency_analysis)
)

print(
    "Unique X-chromosome variants:",
    x_frequency_analysis["variant"].nunique()
)

x_frequency_analysis = x_frequency_analysis[
    x_frequency_analysis["sex"].isin(
        [
            "XX",
            "XY",
        ]
    )
].copy()

x_model_data = x_frequency_analysis[
    x_frequency_analysis["variant"].isin(
        included_variant_ids
    )
    & (
        x_frequency_analysis[
            "is_par"
        ] == False
    )
].copy()

x_flagged_common = x_frequency_analysis[
    x_frequency_analysis["variant"].isin(
        excluded_common_variant_ids
    )
    & (
        x_frequency_analysis[
            "is_par"
        ] == False
    )
].copy()

x_allele_burden = (
    x_model_data
    .groupby(
        [
            "GeneSymbol",
            "HGNC_ID",
            "PhenotypeList",
            "Disease_ID",
            "MONDO_ID",
            "population",
            "sex",
        ],
        dropna=False,
        as_index=False,
    )
    .agg(
        pathogenic_allele_frequency=(
            "af",
            "sum",
        ),
        number_of_variants=(
            "variant",
            "nunique",
        ),
        minimum_an=(
            "an",
            "min",
        ),
    )
)

x_allele_burden[
    "invalid_q"
] = (
    x_allele_burden[
        "pathogenic_allele_frequency"
    ] > 1
)

print(
    "\nX-chromosome combinations with q > 1:",
    x_allele_burden["invalid_q"].sum()
)

q_x = x_allele_burden[
    "pathogenic_allele_frequency"
]

x_allele_burden[
    "xy_pathogenic_genotype_frequency"
] = np.where(
    x_allele_burden["sex"] == "XY",
    q_x,
    np.nan,
)

x_allele_burden[
    "xx_heterozygous_frequency"
] = np.where(
    x_allele_burden["sex"] == "XX",
    2 * q_x * (1 - q_x),
    np.nan,
)

x_allele_burden[
    "xx_biallelic_genotype_frequency"
] = np.where(
    x_allele_burden["sex"] == "XX",
    q_x ** 2,
    np.nan,
)

x_allele_burden[
    "xx_any_pathogenic_genotype_frequency"
] = np.where(
    x_allele_burden["sex"] == "XX",
    (
        2 * q_x * (1 - q_x)
        + q_x ** 2
    ),
    np.nan,
)

x_allele_burden[
    "xy_pathogenic_genotypes_per_100k"
] = (
    x_allele_burden[
        "xy_pathogenic_genotype_frequency"
    ]
    * PER_100K
)

x_allele_burden[
    "xx_heterozygotes_per_100k"
] = (
    x_allele_burden[
        "xx_heterozygous_frequency"
    ]
    * PER_100K
)

x_allele_burden[
    "xx_biallelic_genotypes_per_100k"
] = (
    x_allele_burden[
        "xx_biallelic_genotype_frequency"
    ]
    * PER_100K
)

x_allele_burden[
    "xx_any_pathogenic_genotype_per_100k"
] = (
    x_allele_burden[
        "xx_any_pathogenic_genotype_frequency"
    ]
    * PER_100K
)



# ============================================================
# Build inheritance-aware X-linked results
# ============================================================

x_prevalence_results = x_allele_burden[
    [
        "GeneSymbol",
        "HGNC_ID",
        "PhenotypeList",
        "Disease_ID",
        "MONDO_ID",
        "population",
        "sex",
        "pathogenic_allele_frequency",
        "number_of_variants",
        "minimum_an",
        "xy_pathogenic_genotypes_per_100k",
        "xx_heterozygotes_per_100k",
        "xx_biallelic_genotypes_per_100k",
        "xx_any_pathogenic_genotype_per_100k",
    ]
].copy()

x_inheritance_evidence = (
    inheritance_model_summary[
        inheritance_model_summary[
            "inheritance_model"
        ].isin(X_LINKED_MODELS)
    ][
        [
            "HGNC_ID",
            "MONDO_ID",
            "inheritance_model",
            "inheritance_label",
            "best_validity",
            "best_validity_rank",
            "validity_evidence",
            "evidence_sources",
            "assertion_count",
            "validity_variation",
        ]
    ]
    .drop_duplicates()
    .copy()
)
# ============================================================
# Build X-linked prevalence/genotype result table
# ============================================================

x_prevalence_results = x_allele_burden[
    [
        "GeneSymbol",
        "HGNC_ID",
        "PhenotypeList",
        "Disease_ID",
        "MONDO_ID",
        "population",
        "sex",
        "pathogenic_allele_frequency",
        "number_of_variants",
        "minimum_an",
        "xy_pathogenic_genotypes_per_100k",
        "xx_heterozygotes_per_100k",
        "xx_biallelic_genotypes_per_100k",
        "xx_any_pathogenic_genotype_per_100k",
    ]
].copy()

x_inheritance_evidence = (
    inheritance_model_summary[
        inheritance_model_summary[
            "inheritance_model"
        ].isin(
            [
                "x_linked_recessive",
                "x_linked_dominant",
                "x_linked_unspecified",
            ]
        )
    ][
        [
            "HGNC_ID",
            "MONDO_ID",
            "inheritance_model",
            "inheritance_label",
            "best_validity",
            "best_validity_rank",
            "validity_evidence",
            "evidence_sources",
            "assertion_count",
            "validity_variation",
        ]
    ]
    .drop_duplicates()
    .copy()
)

x_prevalence_results = (
    x_prevalence_results
    .merge(
        x_inheritance_evidence,
        on=[
            "HGNC_ID",
            "MONDO_ID",
        ],
        how="left",
        validate="m:m",
    )
)

x_prevalence_results[
    "estimated_genetic_prevalence_per_100k"
] = np.nan

x_prevalence_results[
    "prevalence_interpretation"
] = "Genotype frequencies only"

xy_mask = (
    x_prevalence_results["sex"] == "XY"
)

x_prevalence_results.loc[
    xy_mask,
    "estimated_genetic_prevalence_per_100k",
] = x_prevalence_results.loc[
    xy_mask,
    "xy_pathogenic_genotypes_per_100k",
]

x_prevalence_results.loc[
    xy_mask,
    "prevalence_interpretation",
] = "Observed hemizygous pathogenic genotype frequency"

xx_recessive_mask = (
    (x_prevalence_results["sex"] == "XX")
    & (
        x_prevalence_results[
            "inheritance_model"
        ] == "x_linked_recessive"
    )
)

x_prevalence_results.loc[
    xx_recessive_mask,
    "estimated_genetic_prevalence_per_100k",
] = x_prevalence_results.loc[
    xx_recessive_mask,
    "xx_biallelic_genotypes_per_100k",
]

x_prevalence_results.loc[
    xx_recessive_mask,
    "prevalence_interpretation",
] = "Expected biallelic pathogenic genotype frequency"

xx_dominant_mask = (
    (x_prevalence_results["sex"] == "XX")
    & (
        x_prevalence_results[
            "inheritance_model"
        ] == "x_linked_dominant"
    )
)

x_prevalence_results.loc[
    xx_dominant_mask,
    "estimated_genetic_prevalence_per_100k",
] = x_prevalence_results.loc[
    xx_dominant_mask,
    "xx_any_pathogenic_genotype_per_100k",
]

x_prevalence_results.loc[
    xx_dominant_mask,
    "prevalence_interpretation",
] = "Expected frequency carrying at least one pathogenic allele"

no_x_inheritance_evidence = (
    x_prevalence_results[
        "inheritance_model"
    ].isna()
)

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "inheritance_model",
] = "x_linked_unannotated"

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "inheritance_label",
] = "No supporting X-linked assertion"

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "best_validity",
] = "No supporting assertion"

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "best_validity_rank",
] = VALIDITY_RANK[
    "No supporting assertion"
]

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "validity_evidence",
] = "No supporting assertion"

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "evidence_sources",
] = "No supporting source"

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "assertion_count",
] = 0

x_prevalence_results.loc[
    no_x_inheritance_evidence,
    "validity_variation",
] = False

print("\nKDM5C inheritance-aware X-linked example")
print("----------------------------------------")

kdm5c_example = x_prevalence_results[
    (
        x_prevalence_results[
            "GeneSymbol"
        ] == "KDM5C"
    )
    & (
        x_prevalence_results[
            "population"
        ] == "overall"
    )
].copy()

print(
    kdm5c_example[
        [
            "GeneSymbol",
            "PhenotypeList",
            "sex",
            "inheritance_label",
            "pathogenic_allele_frequency",
            "xy_pathogenic_genotypes_per_100k",
            "xx_heterozygotes_per_100k",
            "xx_biallelic_genotypes_per_100k",
            "estimated_genetic_prevalence_per_100k",
            "prevalence_interpretation",
            "best_validity",
            "validity_evidence",
            "evidence_sources",
        ]
    ]
    .to_string(index=False)
)
