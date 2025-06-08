# extract_field_descriptions.py
import yaml
import urllib.request
import pandas as pd
import json # For pretty printing dictionaries
import os

def fetch_and_extract_all_descriptions(output_txt_file="openfda_drug_event_field_descriptions.txt",
                                       output_csv_file="openfda_drug_event_field_descriptions.csv",
                                       output_excel_file="openfda_drug_event_field_descriptions.xlsx"):
    """
    Downloads the drugevent.yaml file from openFDA, extracts descriptions
    for ALL fields, and saves them to text, CSV, and Excel files.
    """
    yaml_url = "https://open.fda.gov/fields/drugevent.yaml"
    print(f"Downloading field descriptions from: {yaml_url}")

    try:
        filehandle, _ = urllib.request.urlretrieve(yaml_url)
        with open(filehandle, "r") as stream:
            attribute_map = yaml.safe_load(stream)
        print("drugevent.yaml downloaded and parsed successfully.")
    except Exception as e:
        print(f"Error downloading or parsing drugevent.yaml: {e}")
        return

    if not attribute_map or 'properties' not in attribute_map:
        print("Could not find 'properties' in the attribute map.")
        return

    extracted_data = []

    # Function to recursively extract field information
    def extract_info(properties_dict, parent_key=""):
        for key, value_dict in properties_dict.items():
            current_key_path = f"{parent_key}.{key}" if parent_key else key
            
            # Ensure value_dict is a dictionary, skip if not
            if not isinstance(value_dict, dict):
                continue

            # Extract basic information - handle None and empty strings properly
            description = value_dict.get('description')
            field_type = value_dict.get('type')
            field_format = value_dict.get('format')
            pattern = value_dict.get('pattern')
            is_exact = value_dict.get('is_exact')
            
            # Clean up all fields - convert None or empty to empty string
            description = description.strip() if description and isinstance(description, str) else ""
            field_type = field_type.strip() if field_type and isinstance(field_type, str) else ""
            field_format = field_format.strip() if field_format and isinstance(field_format, str) else ""
            pattern = pattern.strip() if pattern and isinstance(pattern, str) else ""
            
            # Handle possible_values with different structures
            possible_values_dict = value_dict.get('possible_values')
            possible_values_str = ""
            
            # Only process possible_values if it's not None and not empty
            if possible_values_dict is not None and possible_values_dict != {}:
                try:
                    pv_type = possible_values_dict.get('type', '')
                    if pv_type == 'one_of' and 'value' in possible_values_dict:
                        # Handle one_of type with key-value pairs
                        values = possible_values_dict['value']
                        if isinstance(values, dict) and values:  # Make sure values is not empty
                            formatted_values = []
                            for k, v in values.items():
                                formatted_values.append(f"'{k}': \"{v}\"")
                            possible_values_str = "{\n  " + ",\n  ".join(formatted_values) + "\n}"
                    elif pv_type == 'reference' and 'value' in possible_values_dict:
                        # Handle reference type with name and link
                        ref_info = possible_values_dict['value']
                        if isinstance(ref_info, dict):
                            name = ref_info.get('name', '').strip()
                            link = ref_info.get('link', '').strip()
                            if name or link:  # Only show if there's actual content
                                possible_values_str = f"Reference: {name}\nLink: {link}"
                    elif 'value' in possible_values_dict and possible_values_dict['value']:
                        # Handle other cases where value exists and is not empty
                        possible_values_str = json.dumps(possible_values_dict['value'], indent=2, ensure_ascii=False)
                except (TypeError, AttributeError, json.JSONDecodeError):
                    pass  # Skip malformed possible_values

            # Only add fields that have meaningful data (not empty strings)
            if description or field_type or possible_values_str or field_format or pattern or (is_exact is not None and is_exact != ''):
                row_data = {
                    "Full Field Path": current_key_path,
                    "Description": description,
                    "Type": field_type,
                    "Format": field_format,
                    "Pattern": pattern,
                    "Is Exact": str(is_exact) if is_exact is not None and is_exact != '' else "",
                    "Possible Values": possible_values_str
                }
                extracted_data.append(row_data)

            # If there are nested properties, recurse
            if 'properties' in value_dict and isinstance(value_dict['properties'], dict):
                extract_info(value_dict['properties'], current_key_path)
            # Handle items in arrays (like patient.drug, patient.reaction)
            elif 'items' in value_dict and isinstance(value_dict['items'], dict):
                if 'properties' in value_dict['items'] and isinstance(value_dict['items']['properties'], dict):
                    extract_info(value_dict['items']['properties'], f"{current_key_path}[]")

    print("\nExtracting all field descriptions...")
    # Start extraction from the top-level 'properties'
    if 'properties' in attribute_map and isinstance(attribute_map['properties'], dict):
        extract_info(attribute_map['properties'])
    else:
        print("Top-level 'properties' key not found or not a dictionary in attribute_map.")
        return
    
    if not extracted_data:
        print("No field descriptions were extracted.")
        return

    # Create a DataFrame
    df_descriptions = pd.DataFrame(extracted_data)
    df_descriptions.sort_values(by="Full Field Path", inplace=True)

    # --- Save to Text File ---
    try:
        with open(output_txt_file, "w", encoding="utf-8") as f:
            f.write("OpenFDA Drug Event Field Descriptions\n")
            f.write("=======================================\n\n")
            for index, row in df_descriptions.iterrows():
                f.write(f"Field: {row['Full Field Path']}\n")
                if row['Description']:
                    f.write(f"  Description: {row['Description']}\n")
                if row['Type']:
                    f.write(f"  Type: {row['Type']}\n")
                if row['Format']:
                    f.write(f"  Format: {row['Format']}\n")
                if row['Pattern']:
                    f.write(f"  Pattern: {row['Pattern']}\n")
                if row['Is Exact']:
                    f.write(f"  Is Exact: {row['Is Exact']}\n")
                if row['Possible Values']:
                    f.write(f"  Possible Values:\n{row['Possible Values']}\n")
                f.write("-" * 50 + "\n\n")
        print(f"All field descriptions saved to text file: {output_txt_file}")
    except Exception as e:
        print(f"Error saving to text file {output_txt_file}: {e}")

    # --- Save to CSV File ---
    try:
        def simplify_possible_values_for_csv(val_str):
            if val_str and (val_str.startswith('{') or val_str.startswith('[')):
                try:
                    parsed_json = json.loads(val_str)
                    return json.dumps(parsed_json, ensure_ascii=False)
                except json.JSONDecodeError:
                    return val_str 
            return val_str

        df_descriptions_csv = df_descriptions.copy()
        df_descriptions_csv['Possible Values'] = df_descriptions_csv['Possible Values'].apply(simplify_possible_values_for_csv)
        
        df_descriptions_csv.to_csv(output_csv_file, index=False, encoding="utf-8-sig")
        print(f"All field descriptions saved to CSV file: {output_csv_file}")
    except Exception as e:
        print(f"Error saving to CSV file {output_csv_file}: {e}")

    # --- Save to Excel File ---
    try:
        df_descriptions.to_excel(output_excel_file, index=False, engine='openpyxl')
        print(f"All field descriptions saved to Excel file: {output_excel_file}")
    except Exception as e:
        print(f"Error saving to Excel file {output_excel_file}: {e}")


if __name__ == "__main__":
    data_dir = "../../data/openFDA_drug_event/"
    discription_dir = data_dir+'discription/' 
    try:
        os.mkdir(discription_dir)
    except:
        print(discription_dir+" exists")
    txt_filename = discription_dir+"openfda_drug_event_all_field_descriptions.txt"
    csv_filename = discription_dir+"openfda_drug_event_all_field_descriptions.csv"
    excel_filename = discription_dir+"openfda_drug_event_all_field_descriptions.xlsx"
    
    fetch_and_extract_all_descriptions(output_txt_file=txt_filename, output_csv_file=csv_filename, output_excel_file=excel_filename)
    print(f"\nCheck '{txt_filename}', '{csv_filename}', and '{excel_filename}' for the complete list of field descriptions.")