# check_fda_data_partitions_v3_corrected_parsing.py
import json
import urllib.request
import pandas as pd
import re # Import regular expression module

def check_data_partitions_corrected():
    """
    Downloads the FDA download.json, parses drug event partitions based on
    the observed display_name format, and prints a summary of years and quarters.
    """
    print("Attempting to download download.json from FDA...")
    try:
        filehandle, _ = urllib.request.urlretrieve("https://api.fda.gov/download.json")
        with open(filehandle) as json_file:
            data = json.load(json_file)
        print("download.json downloaded and loaded successfully.")
    except Exception as e:
        print(f"Error downloading or loading download.json: {e}")
        return

    if not (data and \
            'results' in data and \
            'drug' in data['results'] and \
            'event' in data['results']['drug'] and \
            'partitions' in data['results']['drug']['event']):
        print("Could not find 'drug.event.partitions' in the downloaded data or data structure is unexpected.")
        return

    partitions = data["results"]["drug"]["event"]["partitions"]
    if not partitions:
        print("No partitions found in the drug event data.")
        return

    print(f"\nFound {len(partitions)} total data partitions for drug events.")

    partition_summary = []
    for p_info in partitions:
        display_name = p_info.get('display_name', '')
        year = None
        quarter = None

        # *** CORRECTED PARSING LOGIC FOR display_name ***
        # Expected format: "YYYY QX (part X of Y)" e.g., "2004 Q3 (part 1 of 5)"
        # We can use regular expression for more robust parsing
        match = re.match(r"(\d{4})\s+Q(\d)", display_name) # \d{4} for year, \s+ for space, Q\d for quarter
        
        if match:
            year_str = match.group(1)
            quarter_num_str = match.group(2)
            
            if year_str.isdigit():
                year = int(year_str)
            if quarter_num_str.isdigit():
                quarter = f"q{quarter_num_str}" # Format as "qX"
        else:
            # Fallback if display_name format is different or parsing failed
            # Try updated_date as a secondary source for year
            updated_date = p_info.get('updated_date', '')
            if updated_date and isinstance(updated_date, str) and len(updated_date) >= 4:
                if updated_date[:4].isdigit():
                    year = int(updated_date[:4])
                    # Quarter would be 'Unknown' or could be derived if needed
                    quarter = 'Unknown (from updated_date)'


        if year:
            partition_summary.append({'year': year, 'quarter': quarter if quarter else 'Unknown'})
        # else:
            # print(f"DEBUG: Could not parse year/quarter from display_name: '{display_name}' or updated_date.")


    if not partition_summary:
        print("\nCould not extract year/quarter information from any partitions with the new parsing logic.")
        print("Please double-check the 'display_name' format in download.json if issues persist.")
        return

    df_summary = pd.DataFrame(partition_summary)

    print("\n--- Summary of Available Data Partitions (Corrected Parsing) ---")

    unique_years = sorted(df_summary['year'].unique())
    print(f"\nAvailable data covers years: {unique_years}")

    print("\nNumber of unique Quarters recorded per year (from display_name):")
    # Filter out 'Unknown' quarters for a cleaner count if display_name was primary source
    year_quarter_counts = df_summary[df_summary['quarter'] != 'Unknown (from updated_date)'].groupby('year')['quarter'].nunique()
    print(year_quarter_counts)
    
    unknown_quarters_by_year = df_summary[df_summary['quarter'] == 'Unknown (from updated_date)'].groupby('year').size()
    if not unknown_quarters_by_year.empty:
        print("\nNumber of partitions where quarter was 'Unknown (derived from updated_date)':")
        print(unknown_quarters_by_year)


    print("\nNumber of partition files per Year and Quarter:")
    files_per_year_q = df_summary.groupby(['year', 'quarter']).size().reset_index(name='file_counts')
    if not files_per_year_q.empty:
        print(files_per_year_q.to_string())
    else:
        print("No year-quarter file counts to display.")

if __name__ == "__main__":
    check_data_partitions_corrected()