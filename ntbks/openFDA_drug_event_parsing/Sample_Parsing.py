#!/usr/bin/env python
# coding: utf-8

# ## openFDA Drug Event data parsing, processing, and output (SAMPLE VERSION)

# import libraries

# In[1]:


import os
import io
import urllib
import requests
import zipfile
import json
import time
import numpy as np
import pandas as pd
from pandas import json_normalize


# read in api token and put in header for api call

# In[2]:


api_token = pd.read_csv("../../.openFDA.params").api_key.values[0]


# In[3]:


headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {0}".format(api_token),
}


# get openFDA drug event links

# In[4]:


filehandle, _ = urllib.request.urlretrieve("https://api.fda.gov/download.json")


# In[5]:


with open(filehandle) as json_file:
    data = json.load(json_file)


# how mmany records are there?

# In[6]:


data["results"]["drug"]["event"]["total_records"]


# how many files do we have?

# In[7]:


len(data["results"]["drug"]["event"]["partitions"])


# put all files into a list

# In[8]:


drug_event_files_all = [
    x["file"] for x in data["results"]["drug"]["event"]["partitions"]
]

# --- SAMPLE MODIFICATION: Limit number of files to process ---
NUMBER_OF_FILES_TO_PROCESS_SAMPLE = 15  # Approx 150k-200k records
drug_event_files = drug_event_files_all[:NUMBER_OF_FILES_TO_PROCESS_SAMPLE]
print(
    f"Processing a sample of {len(drug_event_files)} files out of {len(drug_event_files_all)} total files."
)
# --- END SAMPLE MODIFICATION ---


# create output directory for SAMPLE data

# In[9]:


# --- SAMPLE MODIFICATION: Change to data_SAMPLE directory ---
data_dir = "../../data_SAMPLE/"
# --- END SAMPLE MODIFICATION ---
try:
    os.makedirs(data_dir, exist_ok=True)  # Use makedirs to create parent dirs if needed
except:
    print(data_dir + " could not be created, or already exists")

out = data_dir + "openFDA_drug_event/"
try:
    os.makedirs(out, exist_ok=True)
except:
    print(out + " could not be created, or already exists")

out_report = out + "report/"
try:
    os.makedirs(out_report, exist_ok=True)
except:
    print(out_report + " could not be created, or already exists")

out_meta = out + "meta/"
try:
    os.makedirs(out_meta, exist_ok=True)
except:
    print(out_meta + " could not be created, or already exists")

out_patient = out + "patient/"
try:
    os.makedirs(out_patient, exist_ok=True)
except:
    print(out_patient + " could not be created, or already exists")

out_patient_drug = out + "patient_drug/"
try:
    os.makedirs(out_patient_drug, exist_ok=True)
except:
    print(out_patient_drug + " could not be created, or already exists")

out_patient_drug_openfda = out + "patient_drug_openfda/"
try:
    os.makedirs(out_patient_drug_openfda, exist_ok=True)
except:
    print(out_patient_drug_openfda + " could not be created, or already exists")

out_patient_drug_openfda_rxcui = out + "patient_drug_openfda_rxcui/"
try:
    os.makedirs(out_patient_drug_openfda_rxcui, exist_ok=True)
except:
    print(out_patient_drug_openfda_rxcui + " could not be created, or already exists")

out_patient_reaction = out + "patient_reaction/"
try:
    os.makedirs(out_patient_reaction, exist_ok=True)
except:
    print(out_patient_reaction + " could not be created, or already exists")


# ## drug event attributes

# ### get attributes

# In[10]:


filehandle, _ = urllib.request.urlretrieve("https://open.fda.gov/fields/drugevent.yaml")


# In[11]:


import yaml

with open(filehandle, "r") as stream:
    try:
        attribute_map = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# In[12]:


attribute_map["properties"]


# ## functions

# ### retrive data from files

# In[13]:


def request_and_generate_data(drug_event_file, headers=headers, stream=True):
    t_download_start = time.time()
    response = requests.get(drug_event_file, headers=headers, stream=True)
    t_download_end = time.time()
    download_time = t_download_end - t_download_start
    download_size_bytes = len(response.content)

    zip_file_object = zipfile.ZipFile(io.BytesIO(response.content))
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)
    content = file.read()
    data = json.loads(content.decode())
    return data, download_time, download_size_bytes


# ### report data formatting/mapping function

# In[14]:


def report_formatter(df):
    attributes_dict = attribute_map["properties"]

    cols = np.intersect1d(list(attributes_dict.keys()), df.columns)

    for col in cols:
        try:
            if attributes_dict[col]["possible_values"]["type"] == "one_of":
                attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
                df[col] = df[col].astype(float)
                df[col] = (
                    df[col]
                    .apply(lambda x: str(int(x)) if (x >= 0) else x)
                    .map(attributes_dict_col)
                )
        except:
            pass
    return df


# ### report primarysource formatting/mapping function

# In[15]:


def primarysource_formatter(df):
    keyword = "primarysource"

    attributes_dict = attribute_map["properties"][keyword]["properties"]

    cols = np.intersect1d(
        list(attributes_dict.keys()), [x.replace(keyword + ".", "") for x in df.columns]
    )

    for col in cols:
        try:
            if attributes_dict[col]["possible_values"]["type"] == "one_of":
                attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
                df[keyword + "." + col] = df[keyword + "." + col].astype(float)
                df[keyword + "." + col] = (
                    df[keyword + "." + col]
                    .apply(lambda x: str(int(x)) if (x >= 0) else x)
                    .map(attributes_dict_col)
                )
        except:
            pass
    return df


# ### report serious formatting/mapping function

# In[16]:


def report_serious_formatter(df):
    attributes_dict = attribute_map["properties"]

    col = "serous"  # Note: This might be a typo in original, should it be 'serious'? Keeping as 'serous' to match original logic.

    try:
        attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
        df[col] = df[col].astype(float)
        df[col] = (
            df[col]
            .apply(lambda x: str(int(x)) if (x >= 0) else x)
            .map(attributes_dict_col)
        )
    except:
        pass
    return df


# ### patient data formatting/mapping function

# In[17]:


def patient_formatter(df):
    attributes_dict = attribute_map["properties"]["patient"]["properties"]

    cols = np.intersect1d(
        list(attributes_dict.keys()), [x.replace("patient.", "") for x in df.columns]
    )

    for col in cols:
        try:
            if attributes_dict[col]["possible_values"]["type"] == "one_of":
                attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
                df["patient." + col] = df["patient." + col].astype(float)
                df["patient." + col] = (
                    df["patient." + col]
                    .apply(lambda x: str(int(x)) if (x >= 0) else x)
                    .map(attributes_dict_col)
                )
        except:
            pass
        if (
            "date" in col
        ):  # This check might be too broad, original was `if 'date' in col:`
            try:
                df["patient." + col] = pd.to_datetime(
                    df["patient." + col], format="%Y%m%d", errors="coerce"
                )
            except:  # Fallback or skip if specific format fails
                try:
                    df["patient." + col] = pd.to_datetime(
                        df["patient." + col],
                        infer_datetime_format=True,
                        errors="coerce",
                    )
                except:
                    pass

    aged = df.copy()
    # Ensure required columns exist before proceeding
    if (
        "patient.patientonsetage" not in aged.columns
        or "patient.patientonsetageunit" not in aged.columns
    ):
        print(
            "Warning: 'patient.patientonsetage' or 'patient.patientonsetageunit' not found in patient_formatter. Skipping age calculation."
        )
        return df.assign(master_age=np.nan)

    aged = aged[["patient.patientonsetage", "patient.patientonsetageunit"]].dropna()

    # Convert to numeric, coercing errors, before attempting string operations
    aged["patient.patientonsetage"] = pd.to_numeric(
        aged["patient.patientonsetage"], errors="coerce"
    )
    aged = aged.dropna(subset=["patient.patientonsetage"])

    year_reports = (
        aged[aged["patient.patientonsetageunit"].astype(str) == "801"]  # Year
    ).index.values
    month_reports = (
        aged[aged["patient.patientonsetageunit"].astype(str) == "802"]  # Month
    ).index.values
    day_reports = (
        aged[aged["patient.patientonsetageunit"].astype(str) == "804"]  # Day
    ).index.values
    decade_reports = (
        aged[
            aged["patient.patientonsetageunit"].astype(str) == "800"
        ]  # Decade - Assuming 800, check mapping
    ).index.values
    week_reports = (
        aged[aged["patient.patientonsetageunit"].astype(str) == "803"]  # Week
    ).index.values
    hour_reports = (
        aged[aged["patient.patientonsetageunit"].astype(str) == "805"]  # Hour
    ).index.values

    aged["master_age"] = np.nan

    if len(year_reports) > 0:
        aged.loc[year_reports, "master_age"] = aged.loc[
            year_reports, "patient.patientonsetage"
        ].astype(float)  # Already float after to_numeric
    if len(month_reports) > 0:
        aged.loc[month_reports, "master_age"] = (
            aged.loc[month_reports, "patient.patientonsetage"].astype(float) / 12.0
        )
    if len(week_reports) > 0:
        aged.loc[
            week_reports, "master_age"
        ] = (  # Corrected calculation for weeks to days then to years
            aged.loc[week_reports, "patient.patientonsetage"].astype(float) * 7 / 365.0
        )
    if len(day_reports) > 0:
        aged.loc[day_reports, "master_age"] = (
            aged.loc[day_reports, "patient.patientonsetage"].astype(float) / 365.0
        )
    if len(decade_reports) > 0:
        aged.loc[decade_reports, "master_age"] = (
            aged.loc[decade_reports, "patient.patientonsetage"].astype(float) * 10.0
        )
    if len(hour_reports) > 0:
        aged.loc[hour_reports, "master_age"] = aged.loc[
            hour_reports, "patient.patientonsetage"
        ].astype(float) / (365.0 * 24.0)

    # Join only the 'master_age' column
    df = df.join(aged[["master_age"]])
    # If 'master_age' wasn't added because aged was empty or for other reasons, ensure the column exists
    if "master_age" not in df.columns:
        df = df.assign(master_age=np.nan)

    return df


# ### parse patient.drug data formatting/mapping function

# #### patient.drug formatting/mapping function

# In[18]:


def patient_drug_formatter(df):
    attributes_dict = attribute_map["properties"]["patient"]["properties"]["drug"][
        "items"
    ]["properties"]

    cols = np.intersect1d(list(attributes_dict.keys()), df.columns)

    for col in cols:
        try:
            if attributes_dict[col]["possible_values"]["type"] == "one_of":
                attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
                # Ensure column is numeric before attempting astype(float) or string ops
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # df[col] = df[col].astype(float) # Not needed if to_numeric worked

                if col == "drugadministrationroute":
                    df[col] = (
                        df[col]
                        .apply(
                            lambda x: "".join(
                                np.repeat("0", 3 - len(str(int(x))))
                            )  # Potential error if x is NaN after coerce
                            + str(int(x))
                            if pd.notna(x) and (x >= 0)  # Check for NaN
                            else x
                        )
                        .astype(str)
                        .str.upper()  # Ensure mapping keys are consistent (e.g. all uppercase)
                        .map(attributes_dict_col)
                    )
                else:
                    df[col] = (
                        df[col]
                        .apply(
                            lambda x: str(int(x)) if pd.notna(x) and (x >= 0) else x
                        )  # Check for NaN
                        .astype(str)
                        .str.upper()  # Ensure mapping keys are consistent
                        .map(attributes_dict_col)
                    )
        except KeyError as e:
            # print(f"KeyError in patient_drug_formatter for column {col}: {e}. This might be due to unmapped values or incorrect data types.")
            pass
        except Exception as e:
            # print(f"An error occurred in patient_drug_formatter for column {col}: {e}")
            pass
    return df


# #### main parser formatting/mapping function

# In[19]:


def parse_patient_drug_data(results):
    dict_name = "patient.drug"
    patientdrugs_list = []  # Renamed to avoid conflict, changed to list

    # Ensure 'safetyreportid' and dict_name columns exist
    if "safetyreportid" not in results.columns or dict_name not in results.columns:
        print(
            f"'{dict_name}' or 'safetyreportid' not in results columns. Skipping patient_drug parsing."
        )
        return pd.DataFrame()

    for reportid in results["safetyreportid"].unique():
        lst = []
        # Use .at for faster access if 'safetyreportid' is index, else .loc
        # Assuming 'safetyreportid' might not be index yet or results is a fresh DataFrame
        try:
            # Ensure we are accessing a Series for .loc[reportid] if results is not indexed by safetyreportid
            if results.index.name == "safetyreportid":
                dict_or_list = results.at[reportid, dict_name]
            else:
                dict_or_list = results.loc[
                    results["safetyreportid"] == reportid, dict_name
                ].values[0]

        except (
            IndexError,
            KeyError,
        ):  # Handle cases where reportid might not be found or dict_name column missing for a row
            # print(f"Could not find data for reportid {reportid} in parse_patient_drug_data.")
            continue

        if isinstance(dict_or_list, dict):  # Changed from type() == to isinstance()
            lst.append(
                dict_or_list
            )  # append instead of extend if it's a single dict representing one drug entry
        elif isinstance(dict_or_list, list):
            lst = dict_or_list
        elif isinstance(
            dict_or_list, np.ndarray
        ):  # This case might be problematic if array structure is not as expected
            if dict_or_list.size > 0 and isinstance(dict_or_list[0], list):
                lst = dict_or_list[0]
            elif dict_or_list.size > 0 and isinstance(dict_or_list[0], dict):
                lst = list(dict_or_list)  # Convert array of dicts to list of dicts
            else:
                # print(f"Unexpected np.ndarray structure for reportid {reportid} in parse_patient_drug_data.")
                pass  # or continue

        for i, l_item in enumerate(lst):  # Renamed l to l_item to avoid confusion
            if not isinstance(l_item, dict):  # Ensure l_item is a dictionary
                # print(f"Skipping non-dict item in drug list for reportid {reportid}.")
                continue

            l_copy = l_item.copy()  # l_copy instead of l

            # Remove 'openfda' key robustly
            l_copy.pop("openfda", None)

            # Create DataFrame from the single drug entry
            # The original code `pd.DataFrame({str(reportid): l_copy}).T` creates a DataFrame where reportid is the index
            # and columns are keys from l_copy. This is fine.
            # Renaming axis and resetting index is also fine.
            patientdrug_df_entry = (
                pd.DataFrame({str(reportid): l_copy})
                .T.rename_axis("safetyreportid")
                .reset_index()
            )
            patientdrug_df_entry["entry"] = i
            patientdrugs_list.append(patientdrug_df_entry)

    if not patientdrugs_list:  # Check if list is empty
        # print("No patient drug data found to concatenate.")
        return pd.DataFrame()  # Return empty DataFrame if no data

    allpatientdrugs_df = pd.concat(
        patientdrugs_list, sort=True
    )  # Renamed allpatientdrugs

    # Filter columns more robustly, ensure 'safetyreportid' and 'entry' are kept
    # Original logic: cols_to_keep = allpatientdrugs_df.columns[[type(x) == str for x in allpatientdrugs_df.columns]]
    # This might unintentionally drop numeric column names if they exist and are valid.
    # Assuming all relevant data columns are strings, or we want to keep string-named columns plus 'safetyreportid' and 'entry'.
    # For safety, let's explicitly keep known essential columns and then apply string filter for others if that's the intent.
    # However, it's more likely that all intended columns from JSON are string keys.

    # A safer approach might be to rely on the known schema or ensure all columns are strings after creation if that's an invariant.
    # For now, sticking to original logic's spirit but with a slightly safer check for string type.
    cols_to_keep = [col for col in allpatientdrugs_df.columns if isinstance(col, str)]

    return patient_drug_formatter(allpatientdrugs_df[cols_to_keep])


# ### patient.drug.openfda formatting/mapping function

# #### main parser formatting/mapping function

# In[20]:


def parse_patient_drug_openfda_data(results):
    dict_name = "patient.drug"
    openfdas_list = []  # Renamed

    if "safetyreportid" not in results.columns or dict_name not in results.columns:
        print(
            f"'{dict_name}' or 'safetyreportid' not in results columns. Skipping patient_drug_openfda parsing."
        )
        return pd.DataFrame()

    for reportid in results["safetyreportid"].unique():
        lst = []
        try:
            if results.index.name == "safetyreportid":
                dict_or_list = results.at[reportid, dict_name]
            else:
                dict_or_list = results.loc[
                    results["safetyreportid"] == reportid, dict_name
                ].values[0]
        except (IndexError, KeyError):
            # print(f"Could not find data for reportid {reportid} in parse_patient_drug_openfda_data.")
            continue

        if isinstance(dict_or_list, dict):
            lst.append(dict_or_list)
        elif isinstance(dict_or_list, list):
            lst = dict_or_list
        elif isinstance(dict_or_list, np.ndarray):
            if dict_or_list.size > 0 and isinstance(dict_or_list[0], list):
                lst = dict_or_list[0]
            elif dict_or_list.size > 0 and isinstance(dict_or_list[0], dict):
                lst = list(dict_or_list)
            else:
                # print(f"Unexpected np.ndarray structure for reportid {reportid} in parse_patient_drug_openfda_data.")
                pass

        for i, l_item in enumerate(lst):
            if (
                not isinstance(l_item, dict)
                or "openfda" not in l_item
                or not isinstance(l_item["openfda"], dict)
            ):
                # print(f"Skipping item without 'openfda' dict for reportid {reportid}, entry {i}.")
                continue
            try:
                # Ensure values in openfda_dict are lists for pd.Series, otherwise concat might behave unexpectedly
                openfda_dict = l_item["openfda"]
                processed_openfda_dict = {}
                for k, v in openfda_dict.items():
                    if not isinstance(v, list):
                        processed_openfda_dict[k] = [v]  # Wrap non-list items in a list
                    else:
                        processed_openfda_dict[k] = v

                if not processed_openfda_dict:  # Skip if openfda dict was empty
                    continue

                openfda_df_entry = (  # Renamed openfda
                    pd.concat(
                        {k: pd.Series(v) for k, v in processed_openfda_dict.items()}
                    )
                    .reset_index()
                    # Original code drops level_1. This is fine if all lists in processed_openfda_dict are same length (often 1)
                    # or if we only care about the first element. If lists can have varying lengths and all elements are important,
                    # this part needs rethinking (e.g., exploding or different normalization).
                    # Assuming for now that level_1 corresponds to an index within those lists and dropping it is intended.
                    .rename(columns={"level_0": "key", 0: "value"})
                )
                # Drop 'level_1' only if it exists
                if "level_1" in openfda_df_entry.columns:
                    openfda_df_entry = openfda_df_entry.drop("level_1", axis=1)

                openfda_df_entry["safetyreportid"] = reportid  # Use reportid directly
                openfda_df_entry["entry"] = i
                openfdas_list.append(openfda_df_entry)
            except Exception as e:
                # print(f"Error processing openfda data for reportid {reportid}, entry {i}: {e}")
                pass

    if not openfdas_list:
        # print("No patient.drug.openfda data found.")
        return pd.DataFrame()

    openfdas_df = pd.concat(openfdas_list, sort=True)  # Renamed

    return openfdas_df


# ### parse patient.reaction data formatting/mapping function

# #### patient.reaction formatter function

# In[21]:


def patient_reactions_formatter(df):
    attributes_dict = attribute_map["properties"]["patient"]["properties"]["reaction"][
        "items"
    ]["properties"]

    cols = np.intersect1d(list(attributes_dict.keys()), df.columns)

    for col in cols:
        try:
            if attributes_dict[col]["possible_values"]["type"] == "one_of":
                attributes_dict_col = attributes_dict[col]["possible_values"]["value"]
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # df[col] = df[col].astype(float) # Not needed after to_numeric
                df[col] = (
                    df[col]
                    .apply(
                        lambda x: str(int(x)) if pd.notna(x) and (x >= 0) else x
                    )  # Check for NaN
                    .astype(str)
                    .str.upper()  # Ensure mapping keys are consistent
                    .map(attributes_dict_col)
                )
        except (
            KeyError
        ):  # Handle cases where a value might not be in attributes_dict_col
            # print(f"Warning: Unmapped value found in column {col} during patient_reactions_formatter.")
            pass  # Keep original value or set to NaN if mapping fails due to unmapped key
        except Exception as e:
            # print(f"Error in patient_reactions_formatter for column {col}: {e}")
            pass
    return df


# #### main parser

# In[22]:


def parse_patient_reaction_data(results):
    dict_name = "patient.reaction"
    allpatientreactions_list = []  # Renamed

    if "safetyreportid" not in results.columns or dict_name not in results.columns:
        print(
            f"'{dict_name}' or 'safetyreportid' not in results columns. Skipping patient_reaction parsing."
        )
        return pd.DataFrame()

    for reportid in results["safetyreportid"].unique():
        lst = []
        try:
            if results.index.name == "safetyreportid":
                dict_or_list = results.at[reportid, dict_name]
            else:
                dict_or_list = results.loc[
                    results["safetyreportid"] == reportid, dict_name
                ].values[0]

        except (IndexError, KeyError):
            # print(f"Could not find data for reportid {reportid} in parse_patient_reaction_data.")
            continue

        if isinstance(dict_or_list, dict):
            lst.append(dict_or_list)
        elif isinstance(dict_or_list, list):
            lst = dict_or_list
        elif isinstance(dict_or_list, np.ndarray):
            if dict_or_list.size > 0 and isinstance(dict_or_list[0], list):
                lst = dict_or_list[0]
            elif dict_or_list.size > 0 and isinstance(dict_or_list[0], dict):
                lst = list(dict_or_list)
            else:
                # print(f"Unexpected np.ndarray structure for reportid {reportid} in parse_patient_reaction_data.")
                pass

        rxs_list_for_reportid = []  # Renamed rxs
        for i, l_item in enumerate(lst):  # Renamed l
            if not isinstance(l_item, dict):  # Ensure l_item is a dictionary
                # print(f"Skipping non-dict item in reaction list for reportid {reportid}.")
                continue

            # Original code: pd.DataFrame(l_item, index=[reportid])
            # This creates a DataFrame where each key in l_item becomes a column,
            # and the values (if scalar) are repeated for the single row index 'reportid'.
            # If values in l_item are lists/arrays, this might not be what's intended unless they are all same length.
            # Given the context, l_item is likely a dict of scalar reaction properties.
            rx_df_entry = (  # Renamed rx
                pd.DataFrame(
                    l_item, index=[str(reportid)]
                )  # Ensure index is string if reportid is string
                .rename_axis("safetyreportid")
                .reset_index()
            )
            rx_df_entry["entry"] = i
            rxs_list_for_reportid.append(rx_df_entry)

        if rxs_list_for_reportid:  # If any reactions were processed for this reportid
            allpatientreactions_list.append(pd.concat(rxs_list_for_reportid, sort=True))

    if not allpatientreactions_list:
        # print("No patient reaction data found.")
        return pd.DataFrame()

    # Concatenate all DataFrames from different report IDs
    final_reactions_df = pd.concat(allpatientreactions_list, sort=True)

    return patient_reactions_formatter(
        final_reactions_df  # Already concatenated
    ).reset_index(drop=True)


# ### main parsing function

# In[23]:


def parsing_main(drug_event_file, file_index, total_files):
    t0 = time.time()

    file_lst = drug_event_file.split("/")[2:]
    out_file = "_".join(file_lst[3:]).split(".")[0]

    progress_percent = (file_index + 1) / total_files * 100
    print(
        f"\nStarting download for {out_file} ({file_index + 1}/{total_files} - {progress_percent:.2f}%)..."
    )

    try:
        data, download_time, download_size_bytes = request_and_generate_data(
            drug_event_file, headers=headers, stream=True
        )
        download_size_mb = download_size_bytes / (1024 * 1024)
        print(
            f"Downloaded {out_file}: Size={download_size_mb:.2f} MB, Time={download_time:.2f} sec."
        )

        # parse metadata
        meta = json_normalize(data["meta"])
        try:
            total_records_in_file = meta.get("results.total", pd.Series([np.nan])).iloc[
                0
            ]  # Ensure Series for iloc
            if pd.notna(total_records_in_file):
                print(f"Total records in {out_file}: {int(total_records_in_file)}")
            else:
                print(
                    f"Could not determine total records from metadata for {out_file}."
                )
        except Exception as e:
            print(f"Error accessing total records from metadata for {out_file}: {e}")

        (meta.to_csv(out_meta + out_file + "_meta.csv.gzip", compression="gzip"))
        del meta

        results_raw = data.get("results")
        if not results_raw:
            print(
                f"No 'results' field in JSON for {out_file}. Skipping further processing."
            )
            return  # Exit this function call for this file

        results = json_normalize(results_raw)
        if results.empty:
            print(
                f"'results' field was empty after normalization for {out_file}. Skipping further processing."
            )
            return

        # Ensure 'safetyreportid' exists, if not, this file is problematic
        if "safetyreportid" not in results.columns:
            print(
                f"'safetyreportid' not found in results for {out_file}. Cannot process this file further."
            )
            return

        # parse and output report data
        # results.index = results["safetyreportid"].values # This might fail if safetyreportid has duplicates or NaNs
        # results = results.rename_axis("safetyreportid") # Better to do this after ensuring index is unique and valid

        # Columns to drop for the 'report' part
        # Check if these columns exist before trying to drop them
        cols_to_drop_for_report = []
        if "patient.drug" in results.columns:
            cols_to_drop_for_report.append("patient.drug")
        if "patient.reaction" in results.columns:
            cols_to_drop_for_report.append("patient.reaction")

        report = results.drop(columns=cols_to_drop_for_report, errors="ignore")

        try:
            # Identify patient-related columns to drop for the main report_df
            patient_cols_in_report = [
                col for col in report.columns if col.startswith("patient.")
            ]

            report_df = (
                primarysource_formatter(
                    report_formatter(
                        report_serious_formatter(report.copy())
                    )  # Pass a copy to avoid modifying 'report'
                )
                .drop(
                    columns=patient_cols_in_report, errors="ignore"
                )  # Drop patient related columns
                .reset_index(drop=True)  # Reset index before saving
            )
            if not report_df.empty:
                (
                    report_df.to_csv(
                        out_report + out_file + "_report.csv.gzip", compression="gzip"
                    )
                )
            del report_df  # Delete the copy
        except Exception as e:  # Catch specific exceptions if possible
            print(f"Could not parse or save report data in {out_file}: {e}")
            # pass # Original code used pass, but printing error is more informative

        # Patient data
        try:
            patient_related_cols = [
                col for col in results.columns if col.startswith("patient.")
            ]
            if not patient_related_cols:
                print(
                    f"No patient.* columns found in results for {out_file}. Skipping patient data parsing."
                )
            else:
                patient_df_raw = results[
                    patient_related_cols + ["safetyreportid"]
                ].copy()  # Include safetyreportid for context

                # Columns to drop for the 'patient' part
                cols_to_drop_for_patient = []
                if "patient.drug" in patient_df_raw.columns:
                    cols_to_drop_for_patient.append("patient.drug")
                if "patient.reaction" in patient_df_raw.columns:
                    cols_to_drop_for_patient.append("patient.reaction")

                patient_df = patient_df_raw.drop(
                    columns=cols_to_drop_for_patient, errors="ignore"
                )

                # patient_formatter expects 'patient.patientonsetage' and 'patient.patientonsetageunit'
                # It also joins 'master_age'. Ensure 'safetyreportid' is preserved or handled.
                # The original reset_index() might drop safetyreportid if it was the index.
                # If 'safetyreportid' from results is used as index for patient_df before formatter, ensure it's handled.

                # Assuming patient_formatter returns a DataFrame that might need safetyreportid re-associated or kept
                # The original code did .reset_index().to_csv(...) which implies 'safetyreportid' might be in columns or is the index.

                # Let's ensure 'safetyreportid' is a column before passing to patient_formatter if it relies on it.
                # Or, if patient_formatter needs it as index, set it.
                # For simplicity, let's assume patient_formatter can handle 'safetyreportid' as a column.
                # The original code `patient_formatter(patient_df).reset_index().to_csv(...)`
                # If patient_df has 'safetyreportid' as index, reset_index() makes it a column.
                # If patient_df has 'safetyreportid' as column, reset_index() does nothing to it if index is default.

                formatted_patient_df = patient_formatter(
                    patient_df
                )  # patient_df already includes 'safetyreportid'

                # Ensure 'safetyreportid' is present before saving. patient_formatter might alter columns.
                # If patient_formatter expects 'safetyreportid' as index, it should be set before calling.
                # If it returns it as index, reset_index() is fine.
                # The current patient_formatter doesn't seem to manipulate the index directly, but joins 'master_age'.

                if not formatted_patient_df.empty:
                    (
                        formatted_patient_df.reset_index(
                            drop=True
                        ).to_csv(  # Reset index for clean CSV
                            out_patient + out_file + "_patient.csv.gzip",
                            compression="gzip",
                        )
                    )
                del patient_df_raw
                del patient_df
                del formatted_patient_df
        except Exception as e:
            print(f"Could not parse or save patient data in {out_file}: {e}")
            # pass

        # patient.drug data
        # Pass 'results' which has 'safetyreportid' as a column
        try:
            patientdrug_df = parse_patient_drug_data(results.copy())  # Pass copy
            if not patientdrug_df.empty:
                (
                    patientdrug_df.reset_index(drop=True).to_csv(
                        out_patient_drug + out_file + "_patient_drug.csv.gzip",
                        compression="gzip",
                    )
                )
            del patientdrug_df
        except Exception as e:
            print(f"Could not parse or save patient.drug data in {out_file}: {e}")
            # pass

        # patient.drug.openfda data
        try:
            openfdas_df = parse_patient_drug_openfda_data(results.copy())  # Pass copy
            if not openfdas_df.empty:
                (
                    openfdas_df.reset_index(
                        drop=True
                    ).to_csv(  # reset_index was already in original
                        out_patient_drug_openfda
                        + out_file
                        + "_patient_drug_openfda.csv.gzip",
                        compression="gzip",
                    )
                )
                # Query might fail if 'key' column doesn't exist or 'rxcui' isn't a value
                if "key" in openfdas_df.columns:
                    rxcui_df = openfdas_df.query('key=="rxcui"')
                    if not rxcui_df.empty:
                        (
                            rxcui_df.to_csv(  # Original did not have reset_index here, might be intentional
                                out_patient_drug_openfda_rxcui
                                + out_file
                                + "_patient_drug_openfda_rxcui.csv.gzip",
                                compression="gzip",
                            )
                        )
                    del rxcui_df
                else:
                    print(
                        f"'key' column not found in openfdas_df for {out_file}, skipping rxcui export."
                    )

            del openfdas_df
        except Exception as e:
            print(
                f"Could not parse or save patient.drug.openfda data in {out_file}: {e}"
            )
            # pass

        # patient.reaction data
        try:
            patientreactions_df = parse_patient_reaction_data(
                results.copy()
            )  # Pass copy, Renamed variable
            if not patientreactions_df.empty:
                (
                    patientreactions_df.reset_index(
                        drop=True
                    ).to_csv(  # reset_index was already in original
                        out_patient_reaction + out_file + "_patient_reaction.csv.gzip",
                        compression="gzip",
                    )
                )
            del patientreactions_df  # Renamed variable
        except Exception as e:
            print(f"Could not parse or save patient.reaction data in {out_file}: {e}")
            # pass

        del results  # Delete the large intermediate DataFrame
        del data  # Delete raw data

        t1 = time.time()
        print(
            f"\nFinished processing {out_file} in {np.round(t1 - t0, 2)} seconds. Total progress: {progress_percent:.2f}%\n"  # Changed to t1-t0, round 2
        )

    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to json data for {out_file}")
        # pass # Original behavior
    except (
        Exception
    ) as e:  # Catch any other unexpected error during the main processing of a file
        print(f"An critical error occurred while processing file {out_file}: {e}")
        # pass # Original behavior


#
# ## main

# In[24]:


from joblib import Parallel, delayed
# from dask import delayed, compute, persist
# from dask.distributed import Client, LocalCluster, progress

n_jobs = 4  # You can adjust this based on your CPU cores for the sample run

# if __name__=='__main__': # Standard practice to wrap main execution
t0_loop = time.time()

total_files_to_process = len(
    drug_event_files
)  # Use the length of the (potentially) sliced list
print(
    f"Starting parallel processing of {total_files_to_process} files with {n_jobs} jobs..."
)

Parallel(n_jobs=n_jobs)(
    delayed(parsing_main)(
        drug_event_file, i, total_files_to_process
    )  # Pass total_files_to_process
    for i, drug_event_file in enumerate(
        drug_event_files
    )  # Iterate over the (potentially) sliced list
)

# cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
# c = Client(cluster)
# results = [delayed(parsing_main)(drug_event_file) for drug_event_file in drug_event_files]
# compute(*results[:1])      # convert to final result when done if desired

t1_loop = time.time()
print(
    "\n"
    + str(np.round(t1_loop - t0_loop, 0))
    + " seconds to parse all selected sample files."
)


# In[ ]:
