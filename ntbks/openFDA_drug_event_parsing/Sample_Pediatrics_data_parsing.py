#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import pandas as pd
import pickle
import os

# --- SAMPLE MODIFICATION: Change input directory ---
data_dir = "../../data_SAMPLE/openFDA_drug_event/"
er_dir = os.path.join(data_dir, "er_tables/")
# --- END SAMPLE MODIFICATION ---


# In[2]:


primarykey = "safetyreportid"


# In[3]:


patient_file_path = os.path.join(er_dir, "patient.csv.gz")
patients = pd.DataFrame()
try:
    patients = pd.read_csv(
        patient_file_path,
        compression="gzip",
        index_col=0,
        dtype={primarykey: "str", "patient_custom_master_age": "float"},
    )
    if primarykey not in patients.columns and patients.index.name == primarykey:
        patients.reset_index(inplace=True)
    if primarykey not in patients.columns:
        print(
            f"Warning: Primary key '{primarykey}' not found in {patient_file_path}. Proceeding with empty patient data."
        )
        patients = pd.DataFrame()

except FileNotFoundError:
    print(
        f"File not found: {patient_file_path}. Please ensure Sample_openFDA_Entity_Relationship_Tables.py has been run and generated this file."
    )
    patients = pd.DataFrame()
except Exception as e:
    print(f"Error reading {patient_file_path}: {e}")
    patients = pd.DataFrame()


# In[4]:
aged = pd.DataFrame()
age_col = "patient_onsetage"

if not patients.empty and age_col in patients.columns:
    if patients[age_col].notnull().any():
        aged = patients[patients[age_col].notnull()].reset_index(drop=True).copy()
    else:
        print(
            f"Column '{age_col}' in patients data is all NaN. Cannot proceed with age-based filtering."
        )
elif patients.empty:
    print("Patients DataFrame is empty. Cannot proceed with age classification.")
elif age_col not in patients.columns:
    print(
        f"Column '{age_col}' not found in patients DataFrame. Cannot proceed with age classification."
    )


# In[5]:
if not aged.empty and age_col in aged.columns:
    current_col = "nichd"
    aged_onsetage_numeric = pd.to_numeric(aged[age_col], errors="coerce")

    neonate = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 0 < float(x) <= (1 / 12)
    )
    infant = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and (1 / 12) < float(x) <= 1
    )
    toddler = aged_onsetage_numeric.apply(lambda x: pd.notnull(x) and 1 < float(x) <= 2)
    echildhood = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 2 < float(x) <= 5
    )
    mchildhood = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 5 < float(x) <= 11
    )
    eadolescence = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 11 < float(x) <= 18
    )
    ladolescence = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 18 < float(x) <= 21
    )

    aged[current_col] = pd.Series(dtype="object")  # Initialize with object dtype
    aged.loc[neonate, current_col] = "term_neonatal"
    aged.loc[infant, current_col] = "infancy"
    aged.loc[toddler, current_col] = "toddler"
    aged.loc[echildhood, current_col] = "early_childhood"
    aged.loc[mchildhood, current_col] = "middle_childhood"
    aged.loc[eadolescence, current_col] = "early_adolescence"
    aged.loc[ladolescence, current_col] = "late_adolescence"
else:
    if aged.empty:
        print("Aged DataFrame is empty. Skipping NICHD classification.")
    elif age_col not in aged.columns:
        print(
            f"'{age_col}' column not in aged DataFrame. Skipping NICHD classification."
        )


# In[6]:
if not aged.empty and age_col in aged.columns:
    current_col = "ich_ema"
    aged_onsetage_numeric = pd.to_numeric(aged[age_col], errors="coerce")

    term_newborn_infants = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 0 < float(x) <= (1 / 12)
    )
    infants_and_toddlers = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and (1 / 12) < float(x) <= 2
    )
    children_ich_ema = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 2 < float(x) <= 11
    )
    adolescents_ich_ema = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 11 < float(x) <= 17
    )

    aged[current_col] = pd.Series(dtype="object")  # Initialize with object dtype
    aged.loc[term_newborn_infants, current_col] = "term_newborn_infants"
    aged.loc[infants_and_toddlers, current_col] = "infants_and_toddlers"
    aged.loc[children_ich_ema, current_col] = "children"
    aged.loc[adolescents_ich_ema, current_col] = "adolescents"
else:
    print(
        "Aged DataFrame is empty or missing age column. Skipping ICH/EMA classification."
    )

# In[7]:

if not aged.empty and age_col in aged.columns:
    current_col = "fda"
    aged_onsetage_numeric = pd.to_numeric(aged[age_col], errors="coerce")

    neonates_fda = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 0 < float(x) < (1 / 12)
    )
    infants_fda = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and (1 / 12) <= float(x) < 2
    )
    children_fda = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 2 <= float(x) < 11
    )
    adolescents_fda = aged_onsetage_numeric.apply(
        lambda x: pd.notnull(x) and 11 <= float(x) < 16
    )

    aged[current_col] = pd.Series(dtype="object")  # Initialize with object dtype
    aged.loc[neonates_fda, current_col] = "neonates"
    aged.loc[infants_fda, current_col] = "infants"
    aged.loc[children_fda, current_col] = "children"
    aged.loc[adolescents_fda, current_col] = "adolescents"
else:
    print("Aged DataFrame is empty or missing age column. Skipping FDA classification.")


# In[8]:

pediatric_patients = pd.DataFrame()
if not aged.empty and "nichd" in aged.columns:
    pediatric_patients = aged.dropna(subset=["nichd"]).reset_index(drop=True)
    # print(f"Shape of pediatric_patients: {pediatric_patients.shape}")
    # if not pediatric_patients.empty:
    # print(pediatric_patients.head())
else:
    print(
        "Cannot create pediatric_patients DataFrame as 'aged' is empty or 'nichd' column is missing."
    )


# In[9]:

if "patients" in globals() and isinstance(patients, pd.DataFrame):
    del patients
if "aged" in globals() and isinstance(aged, pd.DataFrame):
    del aged


# In[10]:

# if not pediatric_patients.empty:
#     print("Pediatric patients head after initial processing:")
#     print(pediatric_patients.head())
# else:
#     print("pediatric_patients DataFrame is empty after initial processing.")


# In[11]:

report_file_path = os.path.join(er_dir, "report.csv.gz")
report = pd.DataFrame()
try:
    report = pd.read_csv(
        report_file_path, compression="gzip", dtype={"safetyreportid": "str"}
    )
    if primarykey not in report.columns and report.index.name == primarykey:
        report.reset_index(inplace=True)
    if primarykey not in report.columns:
        print(f"Warning: Primary key '{primarykey}' not found in {report_file_path}.")
        report = pd.DataFrame()
    # if not report.empty:
    #     print("Report data head:")
    #     print(report.head())
    # else:
    #     print(f"File {report_file_path} is empty.")
except FileNotFoundError:
    print(f"File not found: {report_file_path}.")
except Exception as e:
    print(f"Error reading {report_file_path}: {e}")
    report = pd.DataFrame()


# In[12]:

pediatric_patients_report = pd.DataFrame()
if not pediatric_patients.empty and not report.empty:
    if primarykey in pediatric_patients.columns and primarykey in report.columns:
        # Ensure PK is string for merging
        pediatric_patients[primarykey] = pediatric_patients[primarykey].astype(str)
        report[primarykey] = report[primarykey].astype(str)

        pediatric_patients_report = pd.merge(
            pediatric_patients, report, on=primarykey, how="inner"
        )
        # print(f"Shape of pediatric_patients_report: {pediatric_patients_report.shape}")
    else:
        print(
            f"Primary key '{primarykey}' not found in one or both DataFrames (pediatric_patients, report) for merge."
        )
# else:
#     if pediatric_patients.empty:
#         print("pediatric_patients DataFrame is empty. Skipping merge with report.")
#     if report.empty:
#         print("report DataFrame is empty. Skipping merge.")


# In[13]:

if "pediatric_patients" in globals() and isinstance(pediatric_patients, pd.DataFrame):
    del pediatric_patients
if "report" in globals() and isinstance(report, pd.DataFrame):
    del report


# In[14]:

report_serious_file_path = os.path.join(er_dir, "report_serious.csv.gz")
report_serious = pd.DataFrame()
try:
    report_serious = pd.read_csv(
        report_serious_file_path, compression="gzip", dtype={primarykey: "str"}
    )
    if (
        primarykey not in report_serious.columns
        and report_serious.index.name == primarykey
    ):
        report_serious.reset_index(inplace=True)
    if primarykey not in report_serious.columns:
        print(
            f"Warning: Primary key '{primarykey}' not found in {report_serious_file_path}."
        )
        report_serious = pd.DataFrame()
    # if not report_serious.empty:
    #     print("Report serious head:")
    #     print(report_serious.head())
except FileNotFoundError:
    print(
        f"File not found: {report_serious_file_path} (this might be expected for sample run)."
    )
except Exception as e:
    print(f"Error reading {report_serious_file_path}: {e}")
    report_serious = pd.DataFrame()


# In[15]:

pediatric_patients_report_serious = pd.DataFrame()
if not pediatric_patients_report.empty and not report_serious.empty:
    if (
        primarykey in pediatric_patients_report.columns
        and primarykey in report_serious.columns
    ):
        pediatric_patients_report[primarykey] = pediatric_patients_report[
            primarykey
        ].astype(str)
        report_serious[primarykey] = report_serious[primarykey].astype(str)

        pediatric_patients_report_serious = pd.merge(
            pediatric_patients_report, report_serious, on=primarykey, how="inner"
        )
        # print(f"Shape of pediatric_patients_report_serious: {pediatric_patients_report_serious.shape}")
    else:
        print(f"Primary key '{primarykey}' not found for merging with report_serious.")
# elif pediatric_patients_report.empty:
# print("pediatric_patients_report is empty. Skipping merge with report_serious.")
# elif report_serious.empty:
# print("report_serious DataFrame is empty. Skipping merge.")


# In[16]:

# if not pediatric_patients_report_serious.empty:
#     print("Pediatric patients report serious head:")
#     print(pediatric_patients_report_serious.head())
# else:
#     print("pediatric_patients_report_serious is empty.")


# In[17]:

if "report_serious" in globals() and isinstance(report_serious, pd.DataFrame):
    del report_serious
if "pediatric_patients_report" in globals() and isinstance(
    pediatric_patients_report, pd.DataFrame
):
    del pediatric_patients_report


# In[18]:

reporter_file_path = os.path.join(er_dir, "reporter.csv.gz")
reporter = pd.DataFrame()
try:
    reporter = pd.read_csv(
        reporter_file_path, compression="gzip", dtype={primarykey: "str"}
    )
    if primarykey not in reporter.columns and reporter.index.name == primarykey:
        reporter.reset_index(inplace=True)
    if primarykey not in reporter.columns:
        print(f"Warning: Primary key '{primarykey}' not found in {reporter_file_path}.")
        reporter = pd.DataFrame()
    # if not reporter.empty:
    #     print("Reporter data head:")
    #     print(reporter.head())
except FileNotFoundError:
    print(f"File not found: {reporter_file_path} (expected for sample).")
except Exception as e:
    print(f"Error reading {reporter_file_path}: {e}")
    reporter = pd.DataFrame()

# In[19]:

pediatric_patients_report_serious_reporter = pd.DataFrame()
if not pediatric_patients_report_serious.empty and not reporter.empty:
    if (
        primarykey in pediatric_patients_report_serious.columns
        and primarykey in reporter.columns
    ):
        pediatric_patients_report_serious[primarykey] = (
            pediatric_patients_report_serious[primarykey].astype(str)
        )
        reporter[primarykey] = reporter[primarykey].astype(str)

        pediatric_patients_report_serious_reporter = pd.merge(
            pediatric_patients_report_serious, reporter, on=primarykey, how="inner"
        )
        # print(f"Shape of pediatric_patients_report_serious_reporter: {pediatric_patients_report_serious_reporter.shape}")
    else:
        print(f"Primary key '{primarykey}' not found for merging with reporter.")
# elif pediatric_patients_report_serious.empty:
#     print("pediatric_patients_report_serious is empty. Skipping merge with reporter.")
# elif reporter.empty:
#     print("reporter DataFrame is empty. Skipping merge.")


# In[20]:

# if not pediatric_patients_report_serious_reporter.empty:
#     print("Pediatric patients report serious reporter head:")
#     print(pediatric_patients_report_serious_reporter.head())
# else:
#     print("pediatric_patients_report_serious_reporter is empty.")


# In[21]:

# if not pediatric_patients_report_serious_reporter.empty:
#     pediatric_patients_report_serious_reporter.info()
# else:
#     print("pediatric_patients_report_serious_reporter is empty. Cannot display info.")


# In[22]:

if "reporter" in globals() and isinstance(reporter, pd.DataFrame):
    del reporter


# In[23]:

if "pediatric_patients_report_serious" in globals() and isinstance(
    pediatric_patients_report_serious, pd.DataFrame
):
    del pediatric_patients_report_serious


# In[24]:

# Path corrected to be relative to workspace root via ../../
output_path_main = os.path.join(
    "../../data_SAMPLE", "pediatric_patients_report_serious_reporter.csv.gz"
)
if not pediatric_patients_report_serious_reporter.empty:
    try:
        os.makedirs(os.path.dirname(output_path_main), exist_ok=True)
        pediatric_patients_report_serious_reporter.to_csv(
            output_path_main, compression="gzip", index=False
        )
        # print(f"Saved main pediatric data to {output_path_main}")
    except Exception as e:
        print(f"Error saving main pediatric data: {e}")
# else:
# print("pediatric_patients_report_serious_reporter is empty. Skipping save.")


# In[25]:

ped_reports = np.array([])
if (
    not pediatric_patients_report_serious_reporter.empty
    and primarykey in pediatric_patients_report_serious_reporter.columns
):
    ped_reports = (
        pediatric_patients_report_serious_reporter[primarykey].astype(str).unique()
    )
    # print(f"Number of unique pediatric reports: {len(ped_reports)}")
# else:
# print("Cannot extract unique pediatric reports as DataFrame is empty or missing primary key.")


# In[26]:

# This reloading step might be redundant if the DF is kept in memory, but kept for consistency with original notebook structure
# pediatric_patients_report_serious_reporter_reloaded = pd.DataFrame()
# if os.path.exists(output_path_main):
#     try:
#         pediatric_patients_report_serious_reporter_reloaded = pd.read_csv(output_path_main, compression="gzip", dtype={primarykey: "str"})
#         if not pediatric_patients_report_serious_reporter_reloaded.empty:
#             # print("Reloaded main pediatric data:")
#             # print(pediatric_patients_report_serious_reporter_reloaded.head())
#             pediatric_patients_report_serious_reporter = pediatric_patients_report_serious_reporter_reloaded # Reassign
#     except Exception as e:
#         print(f"Error reloading main pediatric data: {e}")


# In[27]:

pediatric_standard_drugs_atc = pd.DataFrame()
standard_drugs_atc_path = os.path.join(er_dir, "standard_drugs_atc.csv.gz")
try:
    if ped_reports.size > 0 and os.path.exists(standard_drugs_atc_path):
        temp_df = pd.read_csv(
            standard_drugs_atc_path, compression="gzip", dtype={primarykey: "str"}
        )
        if primarykey not in temp_df.columns and temp_df.index.name == primarykey:
            temp_df.reset_index(inplace=True)

        if not temp_df.empty and primarykey in temp_df.columns:
            pediatric_standard_drugs_atc = temp_df[
                temp_df[primarykey].astype(str).isin(ped_reports)
            ].copy()
            if not pediatric_standard_drugs_atc.empty:
                pediatric_standard_drugs_atc[primarykey] = pediatric_standard_drugs_atc[
                    primarykey
                ].astype(str)
                if "ATC_concept_id" in pediatric_standard_drugs_atc.columns:
                    pediatric_standard_drugs_atc["ATC_concept_id"] = pd.to_numeric(
                        pediatric_standard_drugs_atc["ATC_concept_id"], errors="coerce"
                    )
                    pediatric_standard_drugs_atc = pediatric_standard_drugs_atc.dropna(
                        subset=["ATC_concept_id"]
                    )
                    if not pediatric_standard_drugs_atc.empty:
                        pediatric_standard_drugs_atc["ATC_concept_id"] = (
                            pediatric_standard_drugs_atc["ATC_concept_id"].astype(int)
                        )
                        # print("Pediatric standard drugs ATC head:")
                        # print(pediatric_standard_drugs_atc.head())
    # elif not os.path.exists(standard_drugs_atc_path):
    # print(f"File not found: {standard_drugs_atc_path} (expected for sample).")
except Exception as e:
    print(f"Error loading pediatric_standard_drugs_atc: {e}")


# In[28]:

pediatric_standard_reactions = pd.DataFrame()
standard_reactions_path = os.path.join(er_dir, "standard_reactions.csv.gz")
try:
    if ped_reports.size > 0 and os.path.exists(standard_reactions_path):
        temp_df = pd.read_csv(
            standard_reactions_path, compression="gzip", dtype={primarykey: "str"}
        )
        if primarykey not in temp_df.columns and temp_df.index.name == primarykey:
            temp_df.reset_index(inplace=True)

        if not temp_df.empty and primarykey in temp_df.columns:
            pediatric_standard_reactions = temp_df[
                temp_df[primarykey].astype(str).isin(ped_reports)
            ].copy()
            if not pediatric_standard_reactions.empty:
                pediatric_standard_reactions[primarykey] = pediatric_standard_reactions[
                    primarykey
                ].astype(str)
                if "MedDRA_concept_id" in pediatric_standard_reactions.columns:
                    pediatric_standard_reactions["MedDRA_concept_id"] = pd.to_numeric(
                        pediatric_standard_reactions["MedDRA_concept_id"],
                        errors="coerce",
                    )
                    pediatric_standard_reactions = pediatric_standard_reactions.dropna(
                        subset=["MedDRA_concept_id"]
                    )
                    if not pediatric_standard_reactions.empty:
                        pediatric_standard_reactions["MedDRA_concept_id"] = (
                            pediatric_standard_reactions["MedDRA_concept_id"].astype(
                                int
                            )
                        )
                        # print("Pediatric standard reactions head:")
                        # print(pediatric_standard_reactions.head())
    # elif not os.path.exists(standard_reactions_path):
    # print(f"File not found: {standard_reactions_path} (expected for sample).")
except Exception as e:
    print(f"Error loading pediatric_standard_reactions: {e}")


# In[29]:

# if not pediatric_patients_report_serious_reporter.empty: print(pediatric_patients_report_serious_reporter.head())
# if not pediatric_standard_drugs_atc.empty: print(pediatric_standard_drugs_atc.head())
# if not pediatric_standard_reactions.empty: print(pediatric_standard_reactions.head())


# In[30]:

# intersect_count = 0
# if (
#     not pediatric_standard_drugs_atc.empty and not pediatric_standard_reactions.empty and
#     primarykey in pediatric_standard_drugs_atc.columns and primarykey in pediatric_standard_reactions.columns
# ):
#     intersect_count = len(
#         np.intersect1d(
#             pediatric_standard_drugs_atc[primarykey].astype(str).unique(),
#             pediatric_standard_reactions[primarykey].astype(str).unique(),
#         )
#     )
#     print(f"Intersection count of safetyreportids in ATC and Reactions: {intersect_count}")


# In[31]:

# This is the DataFrame that was previously undefined.
pediatric_patients_report_serious_reporter_drugs_reactions = pd.DataFrame()

if (
    not pediatric_patients_report_serious_reporter.empty
    and not pediatric_standard_drugs_atc.empty
    and not pediatric_standard_reactions.empty
    and primarykey in pediatric_patients_report_serious_reporter.columns
    and primarykey in pediatric_standard_drugs_atc.columns
    and "ATC_concept_id" in pediatric_standard_drugs_atc.columns
    and primarykey in pediatric_standard_reactions.columns
    and "MedDRA_concept_id" in pediatric_standard_reactions.columns
):
    try:
        # Ensure primarykey is string for all merges/joins
        ppsr_df = pediatric_patients_report_serious_reporter.copy()
        ppsr_df[primarykey] = ppsr_df[primarykey].astype(str)

        psda_df = pediatric_standard_drugs_atc.copy()
        psda_df[primarykey] = psda_df[primarykey].astype(str)

        psr_df = pediatric_standard_reactions.copy()
        psr_df[primarykey] = psr_df[primarykey].astype(str)

        # Merge ppsr with drugs_atc
        df_merged_temp = pd.merge(ppsr_df, psda_df, on=primarykey, how="inner")
        df_merged_temp = df_merged_temp.dropna(subset=["ATC_concept_id"])

        # Merge the result with reactions
        # Need to handle potential duplicate columns from previous merge if any (other than PK)
        # For safety, select specific columns or handle suffixes if overlap exists beyond PK
        cols_to_join_from_reactions = [primarykey, "MedDRA_concept_id"]
        # Add other reaction columns if needed, e.g. MedDRA_concept_code, MedDRA_preferred_term
        if "MedDRA_concept_code" in psr_df.columns:
            cols_to_join_from_reactions.append("MedDRA_concept_code")
        if "MedDRA_preferred_term" in psr_df.columns:
            cols_to_join_from_reactions.append("MedDRA_preferred_term")

        df_merged_temp = pd.merge(
            df_merged_temp,
            psr_df[cols_to_join_from_reactions],
            on=primarykey,
            how="inner",
        )
        df_merged_temp = df_merged_temp.dropna(subset=["MedDRA_concept_id"])

        pediatric_patients_report_serious_reporter_drugs_reactions = (
            df_merged_temp.reset_index(drop=True)
        )  # drop=True if old index is not needed

        if not pediatric_patients_report_serious_reporter_drugs_reactions.empty:
            pediatric_patients_report_serious_reporter_drugs_reactions = pediatric_patients_report_serious_reporter_drugs_reactions.reindex(
                columns=sorted(
                    pediatric_patients_report_serious_reporter_drugs_reactions.columns
                )
            )
            for col_name in ["ATC_concept_id", "MedDRA_concept_id"]:
                if (
                    col_name
                    in pediatric_patients_report_serious_reporter_drugs_reactions.columns
                ):
                    s = pd.to_numeric(
                        pediatric_patients_report_serious_reporter_drugs_reactions[
                            col_name
                        ],
                        errors="coerce",
                    ).dropna()
                    if not s.empty:
                        pediatric_patients_report_serious_reporter_drugs_reactions[
                            col_name
                        ] = s.astype(int)
                    else:
                        pediatric_patients_report_serious_reporter_drugs_reactions = pediatric_patients_report_serious_reporter_drugs_reactions.drop(
                            columns=[col_name]
                        )

            if (
                "MedDRA_concept_code"
                in pediatric_patients_report_serious_reporter_drugs_reactions.columns
            ):
                s_code = pd.to_numeric(
                    pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_code,
                    errors="coerce",
                ).dropna()
                if not s_code.empty:
                    pediatric_patients_report_serious_reporter_drugs_reactions.MedDRA_concept_code = s_code.astype(
                        int
                    )  # Removed .copy()
                else:
                    pediatric_patients_report_serious_reporter_drugs_reactions = (
                        pediatric_patients_report_serious_reporter_drugs_reactions.drop(
                            columns=["MedDRA_concept_code"]
                        )
                    )

            # print(f"Shape of final merged data: {pediatric_patients_report_serious_reporter_drugs_reactions.shape}")
            # if not pediatric_patients_report_serious_reporter_drugs_reactions.empty:
            #     print(pediatric_patients_report_serious_reporter_drugs_reactions.head())
    except Exception as e:
        print(f"Error during merging drugs and reactions: {e}")
# else:
# print("One or more required DataFrames for drug/reaction merge are empty or missing key columns.")


# In[32]:

# Path corrected
output_path_final = os.path.join(
    "../../data_SAMPLE",
    "pediatric_patients_report_serious_reporter_drugs_reactions.csv.gz",
)
if not pediatric_patients_report_serious_reporter_drugs_reactions.empty:
    try:
        os.makedirs(os.path.dirname(output_path_final), exist_ok=True)
        pediatric_patients_report_serious_reporter_drugs_reactions.to_csv(
            output_path_final, compression="gzip", index=False
        )
        # print(f"Saved final merged pediatric data to {output_path_final}")
    except Exception as e:
        print(f"Error saving final merged pediatric data: {e}")
# else:
# print("Final merged pediatric DataFrame is empty. Skipping save.")


# In[33]:

if "pediatric_patients_report_serious_reporter" in globals() and isinstance(
    pediatric_patients_report_serious_reporter, pd.DataFrame
):
    del pediatric_patients_report_serious_reporter


# In[34]:

pediatric_standard_drugs = pd.DataFrame()
standard_drugs_path = os.path.join(er_dir, "standard_drugs.csv.gz")
try:
    if ped_reports.size > 0 and os.path.exists(standard_drugs_path):
        temp_df = pd.read_csv(
            standard_drugs_path, compression="gzip", dtype={primarykey: "str"}
        )
        if primarykey not in temp_df.columns and temp_df.index.name == primarykey:
            temp_df.reset_index(inplace=True)

        if not temp_df.empty and primarykey in temp_df.columns:
            pediatric_standard_drugs = temp_df[
                temp_df[primarykey].astype(str).isin(ped_reports)
            ].copy()
            if not pediatric_standard_drugs.empty:
                pediatric_standard_drugs[primarykey] = pediatric_standard_drugs[
                    primarykey
                ].astype(str)
                if "RxNorm_concept_id" in pediatric_standard_drugs.columns:
                    pediatric_standard_drugs["RxNorm_concept_id"] = pd.to_numeric(
                        pediatric_standard_drugs["RxNorm_concept_id"], errors="coerce"
                    )
                    pediatric_standard_drugs = pediatric_standard_drugs.dropna(
                        subset=["RxNorm_concept_id"]
                    )
                    if not pediatric_standard_drugs.empty:
                        pediatric_standard_drugs["RxNorm_concept_id"] = (
                            pediatric_standard_drugs["RxNorm_concept_id"].astype(int)
                        )
                        # print("Pediatric standard drugs head:")
                        # print(pediatric_standard_drugs.head())
    # elif not os.path.exists(standard_drugs_path):
    # print(f"File not found: {standard_drugs_path} (expected for sample).")
except Exception as e:
    print(f"Error loading pediatric_standard_drugs: {e}")


# In[35]:

# Path corrected
rxfiles_dir = "../../RxNorm_relationships_tables/"
rxfile_dict = {}
if os.path.exists(rxfiles_dir) and os.path.isdir(rxfiles_dir):
    rxfiles_list = os.listdir(rxfiles_dir)
    if rxfiles_list:
        for rxfile_item in rxfiles_list:
            if rxfile_item.endswith(
                ".csv"
            ):  # Assuming standard_drug_brands might be CSV not RRF
                key = rxfile_item.split(".")[0]
                try:
                    rxfile_dict[key] = pd.read_csv(
                        os.path.join(rxfiles_dir, rxfile_item), engine="c", index_col=0
                    )
                except Exception as e:
                    print(f"Error reading RxNorm file {rxfile_item}: {e}")
# else:
# print(f"RxNorm relationships directory not found or empty: {rxfiles_dir}.")


# In[36]:

tobrand = []
if rxfile_dict:
    for rxfile_key_item in rxfile_dict.keys():
        if (
            "concept_class_id_2" in rxfile_dict[rxfile_key_item].columns
        ):  # Check column name
            tobrand.append(
                rxfile_dict[rxfile_key_item].query('concept_class_id_2=="Brand Name"')
            )
# else: print("rxfile_dict is empty. Skipping 'tobrand' creation.")


# In[37]:

m = pd.DataFrame()
if not pediatric_standard_drugs.empty and tobrand:
    concatenated_tobrand = pd.concat(tobrand, sort=False) if tobrand else pd.DataFrame()

    if (
        not concatenated_tobrand.empty
        and "concept_id_1" in concatenated_tobrand.columns
    ):
        # Ensure RxNorm_concept_id column exists and is of a suitable type for merging
        if "RxNorm_concept_id" in pediatric_standard_drugs.columns:
            # Convert both merge keys to a common type, e.g., int64, after handling NaNs
            pediatric_standard_drugs["RxNorm_concept_id_clean"] = pd.to_numeric(
                pediatric_standard_drugs["RxNorm_concept_id"], errors="coerce"
            )
            concatenated_tobrand["concept_id_1_clean"] = pd.to_numeric(
                concatenated_tobrand["concept_id_1"], errors="coerce"
            )

            pediatric_standard_drugs_cleaned = pediatric_standard_drugs.dropna(
                subset=["RxNorm_concept_id_clean"]
            )
            concatenated_tobrand_cleaned = concatenated_tobrand.dropna(
                subset=["concept_id_1_clean"]
            )

            if (
                not pediatric_standard_drugs_cleaned.empty
                and not concatenated_tobrand_cleaned.empty
            ):
                m = pd.merge(
                    pediatric_standard_drugs_cleaned,
                    concatenated_tobrand_cleaned,
                    left_on="RxNorm_concept_id_clean",
                    right_on="concept_id_1_clean",
                    how="inner",  # Ensure we only keep matches
                )
                # print(f"Unique {primarykey}s after merging with RxNorm brands: {m[primarykey].nunique() if primarykey in m.columns else 'PK missing'}")
            # Drop temporary columns
            if "RxNorm_concept_id_clean" in pediatric_standard_drugs.columns:
                pediatric_standard_drugs.drop(
                    columns=["RxNorm_concept_id_clean"], inplace=True
                )
            if "concept_id_1_clean" in concatenated_tobrand.columns:
                concatenated_tobrand.drop(columns=["concept_id_1_clean"], inplace=True)
            if "RxNorm_concept_id_clean" in m.columns:
                m.drop(
                    columns=["RxNorm_concept_id_clean"], inplace=True, errors="ignore"
                )
            if "concept_id_1_clean" in m.columns:
                m.drop(columns=["concept_id_1_clean"], inplace=True, errors="ignore")


# In[38]:

m_renamed = pd.DataFrame()
if not m.empty:
    # Columns from concatenated_tobrand that we want to rename and keep
    cols_to_rename_and_keep = {
        "concept_class_id_2": "RxNorm_brand_concept_class_id",  # Renamed to avoid clash with potential future merges
        "concept_code_2": "RxNorm_brand_concept_code",
        "concept_name_2": "RxNorm_brand_concept_name",
        "concept_id_2": "RxNorm_brand_concept_id",
    }
    # Select primarykey from 'm' (originating from pediatric_standard_drugs)
    # and the relevant columns from the RxNorm side of the merge.
    # Ensure these columns exist in 'm' before selecting.

    columns_to_select = [primarykey]  # Must have primary key

    original_rxnorm_cols_in_m = [
        col for col in cols_to_rename_and_keep.keys() if col in m.columns
    ]
    columns_to_select.extend(original_rxnorm_cols_in_m)

    # Also keep original RxNorm_concept_id from pediatric_standard_drugs if it's there
    if (
        "RxNorm_concept_id" in m.columns
        and "RxNorm_concept_id" not in columns_to_select
    ):
        columns_to_select.append("RxNorm_concept_id")

    if (
        primarykey in m.columns and original_rxnorm_cols_in_m
    ):  # Check if essential columns for rename are present
        m_renamed = m[columns_to_select].rename(columns=cols_to_rename_and_keep)
        m_renamed = (
            m_renamed.drop_duplicates()
        )  # Drop duplicates after selection and rename
        # print("RxNorm brand data prepared for saving.")
        # if not m_renamed.empty: print(m_renamed.head())


# In[39]:

# Path corrected
output_path_brands = os.path.join(
    "../../data_SAMPLE", "pediatric_patients_report_drug_brands.csv.gz"
)
if not m_renamed.empty:
    try:
        os.makedirs(os.path.dirname(output_path_brands), exist_ok=True)
        m_renamed.to_csv(output_path_brands, compression="gzip", index=False)
        # print(f"Saved pediatric drug brands data to {output_path_brands}")
    except Exception as e:
        print(f"Error saving pediatric drug brands data: {e}")


# In[40]: # This was the start of the section with undefined variables
# The definitions below are based on the context of what a pediatric parsing script would need

# Initialize final dataframe that was previously undefined
pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui = (
    pd.DataFrame()
)

# Logic to populate it:
# This DataFrame is likely the result of merging pediatric_patients_report_serious_reporter
# with patient_drug_openfda_df and patient_drug_openfda_rxcui_df (after they are processed for ER)
# Let's load the ER versions of openfda and rxcui tables

patient_drug_openfda_er_path = os.path.join(er_dir, "patient_drug_openfda.csv.gz")
patient_drug_openfda_er_df = pd.DataFrame()
try:
    patient_drug_openfda_er_df = pd.read_csv(
        patient_drug_openfda_er_path, compression="gzip", dtype={primarykey: "str"}
    )
    if (
        primarykey not in patient_drug_openfda_er_df.columns
        and patient_drug_openfda_er_df.index.name == primarykey
    ):
        patient_drug_openfda_er_df.reset_index(inplace=True)
    if primarykey not in patient_drug_openfda_er_df.columns:
        patient_drug_openfda_er_df = pd.DataFrame()
except FileNotFoundError:
    print(f"File not found: {patient_drug_openfda_er_path}")
except Exception as e:
    print(f"Error reading {patient_drug_openfda_er_path}: {e}")


patient_drug_openfda_rxcui_er_path = os.path.join(
    er_dir, "patient_drug_openfda_rxcui.csv.gz"
)
patient_drug_openfda_rxcui_er_df = pd.DataFrame()
try:
    patient_drug_openfda_rxcui_er_df = pd.read_csv(
        patient_drug_openfda_rxcui_er_path,
        compression="gzip",
        dtype={primarykey: "str"},
    )
    if (
        primarykey not in patient_drug_openfda_rxcui_er_df.columns
        and patient_drug_openfda_rxcui_er_df.index.name == primarykey
    ):
        patient_drug_openfda_rxcui_er_df.reset_index(inplace=True)
    if primarykey not in patient_drug_openfda_rxcui_er_df.columns:
        patient_drug_openfda_rxcui_er_df = pd.DataFrame()
except FileNotFoundError:
    print(f"File not found: {patient_drug_openfda_rxcui_er_path}")
except Exception as e:
    print(f"Error reading {patient_drug_openfda_rxcui_er_path}: {e}")

# Perform merges
# Start with pediatric_patients_report_serious_reporter_drugs_reactions (ppsrdr)
# This df should contain the core patient, report, serious, reporter, standard_drug_atc, standard_reaction info

print(
    f"[DEBUG] Shape of pediatric_patients_report_serious_reporter_drugs_reactions before merging with openFDA: {pediatric_patients_report_serious_reporter_drugs_reactions.shape}"
)
if not pediatric_patients_report_serious_reporter_drugs_reactions.empty:
    print(
        f"[DEBUG] Columns of ppsrdr: {pediatric_patients_report_serious_reporter_drugs_reactions.columns.tolist()}"
    )

temp_merged_df = pediatric_patients_report_serious_reporter_drugs_reactions.copy()

if not temp_merged_df.empty and primarykey in temp_merged_df.columns:
    temp_merged_df[primarykey] = temp_merged_df[primarykey].astype(str)

    if (
        not patient_drug_openfda_er_df.empty
        and primarykey in patient_drug_openfda_er_df.columns
    ):
        print(
            f"[DEBUG] Merging with patient_drug_openfda_er_df (Shape: {patient_drug_openfda_er_df.shape})"
        )
        patient_drug_openfda_er_df[primarykey] = patient_drug_openfda_er_df[
            primarykey
        ].astype(str)
        # Merge with openfda data - this can be tricky due to one-to-many from report to openfda entries for drugs
        # The ER script should have safetyreportid for each openfda entry.
        # A left merge is appropriate to keep all pediatric report-drug-reaction entries
        # and append openfda info where available.
        temp_merged_df = pd.merge(
            temp_merged_df,
            patient_drug_openfda_er_df,
            on=primarykey,
            how="left",
            suffixes=("", "_openfda"),
        )
    else:
        print(
            "[DEBUG] patient_drug_openfda_er_df is empty or missing primary key. Skipping merge."
        )

    if (
        not patient_drug_openfda_rxcui_er_df.empty
        and primarykey in patient_drug_openfda_rxcui_er_df.columns
    ):
        print(
            f"[DEBUG] Merging with patient_drug_openfda_rxcui_er_df (Shape: {patient_drug_openfda_rxcui_er_df.shape})"
        )
        patient_drug_openfda_rxcui_er_df[primarykey] = patient_drug_openfda_rxcui_er_df[
            primarykey
        ].astype(str)
        # Merge with rxcui data - similar to openfda, can be one-to-many
        # RxCUI in openfda_er_df and patient_drug_openfda_rxcui_er_df can be different sources for RxCUIs
        # The merge should be on safetyreportid.
        # If there are 'rxcui' columns in both, suffixes will be needed or a more careful join.
        # Let's assume patient_drug_openfda_rxcui_er_df is the primary source for detailed RxCUI links per report.

        # Check for rxcui column in temp_merged_df (potentially from openfda_er_df merge)
        # to avoid conflict or decide on merging strategy
        if (
            "rxcui" in temp_merged_df.columns
            and "rxcui" in patient_drug_openfda_rxcui_er_df.columns
        ):
            temp_merged_df = pd.merge(
                temp_merged_df,
                patient_drug_openfda_rxcui_er_df,
                on=primarykey,
                how="left",
                suffixes=("_openfda", "_rxcui_list"),
            )
        elif (
            "rxcui" not in temp_merged_df.columns
            and "rxcui" in patient_drug_openfda_rxcui_er_df.columns
        ):
            temp_merged_df = pd.merge(
                temp_merged_df,
                patient_drug_openfda_rxcui_er_df,
                on=primarykey,
                how="left",
            )
        # Else, no rxcui in patient_drug_openfda_rxcui_er_df, no merge needed for it or already handled.
    else:
        print(
            "[DEBUG] patient_drug_openfda_rxcui_er_df is empty or missing primary key. Skipping merge."
        )

    pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui = (
        temp_merged_df.drop_duplicates()
    )
    if not pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.empty:
        pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui = pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.reindex(
            columns=sorted(
                pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.columns
            )
        )
else:
    print(
        "[DEBUG] temp_merged_df (from ppsrdr) is empty or missing primary key. Skipping openFDA merges."
    )


# In[41]: (This corresponds to original In[55] logic)

# --- SAMPLE MODIFICATION: Output paths should be inside data_SAMPLE ---
# Paths are relative to workspace root via ../../
output_dir_ped_reactions = (
    "../../data_SAMPLE/openFDA_drug_event/pediatric_reactions_by_drug/"
)
output_dir_ped_patient_level_data = (
    "../../data_SAMPLE/openFDA_drug_event/pediatric_patient_level_data/"
)
# --- END SAMPLE MODIFICATION ---

try:
    os.makedirs(output_dir_ped_reactions, exist_ok=True)
    os.makedirs(output_dir_ped_patient_level_data, exist_ok=True)
except OSError as error_os_mkdir:
    print(f"Error creating output directories: {error_os_mkdir}")

# Definition of pediatric_reports_lookup_table (previously undefined)
# This table usually contains a mapping of pediatric safetyreportids to age classifications
pediatric_reports_lookup_table = pd.DataFrame()
# It would be derived from the `pediatric_patients_report_serious_reporter` dataframe or similar,
# selecting primarykey and age group columns (nichd, ich_ema, fda)
# For now, initialize it based on available columns if pprsrdr is populated
if not pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.empty:
    cols_for_lookup = [primarykey]
    age_group_cols = [
        "nichd",
        "ich_ema",
        "fda",
        "patient_custom_master_age",
        "patient_onsetage",
        "patient_onsetageunit",
    ]
    for ag_col in age_group_cols:
        if (
            ag_col
            in pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.columns
        ):
            cols_for_lookup.append(ag_col)
    if len(cols_for_lookup) > 1:  # Found at least one age column besides PK
        pediatric_reports_lookup_table = (
            pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui[
                cols_for_lookup
            ].drop_duplicates(subset=[primarykey])
        )


# This loop iterates over unique drugs and saves reactions per drug.
main_df_for_drug_reactions = (
    pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui
)
drug_col_to_use = None

if (
    "ATC_concept_name" in main_df_for_drug_reactions.columns
    and main_df_for_drug_reactions["ATC_concept_name"].notna().any()
):
    drug_col_to_use = "ATC_concept_name"
elif (
    "RxNorm_brand_concept_name" in main_df_for_drug_reactions.columns
    and main_df_for_drug_reactions["RxNorm_brand_concept_name"].notna().any()
):
    drug_col_to_use = "RxNorm_brand_concept_name"

if drug_col_to_use and not main_df_for_drug_reactions.empty:
    print(f"[DEBUG] Using drug column '{drug_col_to_use}' for per-drug reaction files.")
    unique_drugs = main_df_for_drug_reactions[drug_col_to_use].dropna().unique()
    print(
        f"[DEBUG] Found {len(unique_drugs)} unique drugs for reaction file generation."
    )
    for drug_name in unique_drugs:
        if pd.isna(drug_name) or not drug_name:
            continue  # Skip if drug name is NaN or empty
        unique_drug_file_name = (
            str(drug_name).replace(" ", "_").replace("/", "_").replace("\\", "")
        )
        df_to_save = main_df_for_drug_reactions[
            main_df_for_drug_reactions[drug_col_to_use] == drug_name
        ]
        if not df_to_save.empty:
            out_file_path = os.path.join(
                output_dir_ped_reactions, unique_drug_file_name + ".csv.gzip"
            )
            try:
                df_to_save.to_csv(out_file_path, compression="gzip", index=False)
                print(
                    f"[DEBUG] Saved reaction file: {out_file_path} (Shape: {df_to_save.shape})"
                )
            except Exception as e_csv_save:
                print(f"[DEBUG] Error saving {out_file_path}: {e_csv_save}")
        else:
            print(f"[DEBUG] Skipped saving empty reaction file for drug: {drug_name}")
else:
    print(
        "[DEBUG] Skipping per-drug reaction file generation. Conditions not met (drug column missing, or main DF empty)."
    )
    if main_df_for_drug_reactions.empty:
        print("[DEBUG] Reason: main_df_for_drug_reactions is empty.")
    elif not drug_col_to_use:
        print(
            "[DEBUG] Reason: No suitable drug column ('ATC_concept_name' or 'RxNorm_brand_concept_name' with non-NA values) found in main_df_for_drug_reactions."
        )
        if not main_df_for_drug_reactions.empty:
            print(
                f"[DEBUG] Columns in main_df_for_drug_reactions: {main_df_for_drug_reactions.columns.tolist()}"
            )

# In[63]: (Corresponds to original In[63] -> new numbering due to added cells)

pediatric_patient_level_data_output_file = os.path.join(
    output_dir_ped_patient_level_data, "pediatric_patient_level_data.csv.gzip"
)

print(f"[DEBUG] Attempting to save pediatric_patient_level_data.csv.gz")
print(
    f"[DEBUG] Shape of pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui: {pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.shape}"
)
if not pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.empty:
    print(
        f"[DEBUG] Columns: {pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.columns.tolist()}"
    )
    try:
        pediatric_patients_report_serious_reporter_reactions_drugs_openfda_rxcui.to_csv(
            pediatric_patient_level_data_output_file,
            compression="gzip",
            index=False,
        )
        print(f"[DEBUG] Saved: {pediatric_patient_level_data_output_file}")
    except Exception as e_final_save:
        print(f"[DEBUG] Error saving final pediatric data: {e_final_save}")
else:
    print(
        "[DEBUG] Final pediatric DataFrame (pprsr_drugs_openfda_rxcui) is empty. Skipping save of pediatric_patient_level_data.csv.gz"
    )


# In[64]: (Corresponds to original In[64])

pediatric_reports_lookup_table_output_file = os.path.join(
    output_dir_ped_patient_level_data, "pediatric_reports_lookup_table.csv.gzip"
)
print(f"[DEBUG] Attempting to save pediatric_reports_lookup_table.csv.gz")
print(
    f"[DEBUG] Shape of pediatric_reports_lookup_table: {pediatric_reports_lookup_table.shape}"
)
if not pediatric_reports_lookup_table.empty:
    print(f"[DEBUG] Columns: {pediatric_reports_lookup_table.columns.tolist()}")
    try:
        pediatric_reports_lookup_table.to_csv(
            pediatric_reports_lookup_table_output_file,
            compression="gzip",
            index=False,
        )
        print(f"[DEBUG] Saved: {pediatric_reports_lookup_table_output_file}")
    except Exception as e_lookup_save:
        print(f"[DEBUG] Error saving pediatric reports lookup table: {e_lookup_save}")
else:
    print(
        "[DEBUG] Pediatric reports lookup table is empty. Skipping save of pediatric_reports_lookup_table.csv.gz"
    )

print("Sample Pediatrics data parsing script finished.")
