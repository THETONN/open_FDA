#!/usr/bin/env python
# coding: utf-8

# ## OpenFDA Drug Event Data Consolidation and EDA (SAMPLE VERSION)

import os
import glob
import pandas as pd
import numpy as np

# --- Configuration ---
BASE_DATA_DIR = "../../data_SAMPLE/openFDA_drug_event/"
COMBINED_DATA_DIR = os.path.join(BASE_DATA_DIR, "combined_data")

# Data types and their respective subdirectories
DATA_TYPES = {
    "report": "report/",
    "meta": "meta/",
    "patient": "patient/",
    "patient_drug": "patient_drug/",
    "patient_drug_openfda": "patient_drug_openfda/",
    "patient_drug_openfda_rxcui": "patient_drug_openfda_rxcui/",
    "patient_reaction": "patient_reaction/",
}

# Ensure the combined data directory exists
os.makedirs(COMBINED_DATA_DIR, exist_ok=True)
print(f"Combined data will be saved in: {COMBINED_DATA_DIR}")

# --- Helper Functions ---


def consolidate_data(data_type_name, source_subdir):
    """
    Consolidates all .csv.gzip files from a source subdirectory into a single DataFrame
    and saves it to the COMBINED_DATA_DIR.
    """
    source_path = os.path.join(BASE_DATA_DIR, source_subdir)
    output_file_path = os.path.join(
        COMBINED_DATA_DIR, f"combined_{data_type_name}.csv.gz"
    )

    print(f"\n--- Consolidating data for: {data_type_name} ---")
    print(f"Source directory: {source_path}")

    all_files = glob.glob(os.path.join(source_path, "*.csv.gzip"))

    if not all_files:
        print(f"No files found for {data_type_name} in {source_path}")
        return None

    print(f"Found {len(all_files)} files to consolidate.")

    df_list = []
    for f_index, f_path in enumerate(all_files):
        try:
            print(
                f"Reading file {f_index + 1}/{len(all_files)}: {os.path.basename(f_path)}"
            )
            df_chunk = pd.read_csv(
                f_path,
                compression="gzip",
                low_memory=False,
                dtype={"safetyreportid": str},
            )
            df_list.append(df_chunk)
        except Exception as e:
            print(f"Error reading file {f_path}: {e}")
            continue

    if not df_list:
        print(f"No dataframes were successfully read for {data_type_name}.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Consolidated DataFrame shape for {data_type_name}: {combined_df.shape}")

    try:
        combined_df.to_csv(output_file_path, compression="gzip", index=False)
        print(f"Saved consolidated data to: {output_file_path}")
    except Exception as e:
        print(f"Error saving consolidated file {output_file_path}: {e}")
        return None

    return combined_df


def perform_eda(df, df_name):
    """
    Performs basic Exploratory Data Analysis on a DataFrame.
    """
    if df is None or df.empty:
        print(f"\n--- EDA for {df_name} ---")
        print("DataFrame is empty or None. Skipping EDA.")
        return

    print(f"\n--- EDA for {df_name} (Shape: {df.shape}) ---")

    print("\n### Head ###")
    print(df.head())

    print("\n### Info ###")
    df.info()  # Prints directly

    print("\n### Describe (all columns) ###")
    print(df.describe(include="all").transpose())

    print("\n### Missing Values (%) ###")
    missing_percentage = (df.isnull().sum() / len(df) * 100).sort_values(
        ascending=False
    )
    print(missing_percentage[missing_percentage > 0])

    key_categorical_cols = [
        "safetyreportid",
        "drugcharacterization",
        "reactionoutcome",
        "key",
        "serious",
        "primarysource.qualification",
        "patient.patientsex",
    ]

    print("\n### Value Counts for Selected Categorical Columns ###")
    for col in key_categorical_cols:
        if col in df.columns:
            print(f"\nValue counts for '{col}':")
            top_n = 10
            counts = df[col].value_counts(dropna=False)
            if len(counts) > top_n:
                print(counts.head(top_n))
                print(f"... and {len(counts) - top_n} more unique values.")
            else:
                print(counts)
        else:
            pass


# --- Main Execution ---
if __name__ == "__main__":
    all_combined_dfs = {}
    for type_name, type_subdir in DATA_TYPES.items():
        combined_df = consolidate_data(type_name, type_subdir)
        if combined_df is not None:
            all_combined_dfs[type_name] = combined_df
            perform_eda(combined_df, f"Combined {type_name.capitalize()} Data")
        else:
            print(
                f"Skipping EDA for {type_name} as consolidation failed or produced no data."
            )

    print("\n--- Data Consolidation and EDA Complete ---")

    # Example: Access a combined DataFrame
    # if 'report' in all_combined_dfs:
    #     print("\nExample: Accessing combined 'report' DataFrame head:")
    #     print(all_combined_dfs['report'].head())
