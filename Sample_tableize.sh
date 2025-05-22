#!/bin/bash
# Script to attempt tableization for SAMPLE data
# Assumes 'tabelize' repository is cloned at ../tabelize relative to this script's parent dir
# And vocabularies are set up for tabelize.py to use.

# Define paths
WORKSPACE_ROOT=$(pwd) # Should be /Users/ton/Desktop/openFDA_drug_event_parsing
ER_TABLES_DIR="${WORKSPACE_ROOT}/data_SAMPLE/openFDA_drug_event/er_tables"
# Correctly determine the path to tabelize.py assuming 'tabelize' is a sibling directory to WORKSPACE_ROOT's parent
# e.g. if WORKSPACE_ROOT is /Users/ton/Desktop/openFDA_drug_event_parsing, then parent is /Users/ton/Desktop
# and tabelize.py is at /Users/ton/Desktop/tabelize/tabelize.py
PARENT_DIR_OF_WORKSPACE=$(dirname "$WORKSPACE_ROOT")
TABELIZE_SCRIPT_PATH="${PARENT_DIR_OF_WORKSPACE}/tabelize/tabelize.py"
QUERIES_LOG_FILE="${ER_TABLES_DIR}/sample_tableize_queries.txt"

echo "INFO: Workspace root determined as: $WORKSPACE_ROOT"
echo "INFO: ER Tables directory: $ER_TABLES_DIR"
echo "INFO: Attempting to find tabelize.py at: $TABELIZE_SCRIPT_PATH"
echo "INFO: Queries log will be at: $QUERIES_LOG_FILE"

# Check if tabelize.py exists
if [ ! -f "$TABELIZE_SCRIPT_PATH" ]; then
    echo "ERROR: tabelize.py not found at $TABELIZE_SCRIPT_PATH"
    echo "Please ensure the 'tabelize' repository is cloned into the directory containing this project's parent directory (e.g., /Users/ton/Desktop/tabelize)."
    exit 1
fi

# Check if ER tables directory exists
if [ ! -d "$ER_TABLES_DIR" ]; then
    echo "ERROR: ER tables directory not found at $ER_TABLES_DIR"
    echo "Please run Sample_openFDA_Entity_Relationship_Tables.py first."
    exit 1
fi

echo "INFO: Changing to ER tables directory: $ER_TABLES_DIR"
cd "$ER_TABLES_DIR" || { echo "ERROR: Failed to cd into $ER_TABLES_DIR"; exit 1; }

echo "INFO: Removing old queries log file if it exists: $QUERIES_LOG_FILE"
rm -f "$QUERIES_LOG_FILE"

echo "INFO: Processing files in $(pwd):"
files_to_process=$(ls ./*.csv.gz 2>/dev/null)

if [ -z "$files_to_process" ]; then
    echo "INFO: No .csv.gz files found in $ER_TABLES_DIR to process."
    cd "$WORKSPACE_ROOT" # Go back to original directory
    exit 0
fi

for f in $files_to_process
do
	file_basename=$(basename "$f")
	name_without_ext=$(echo "$file_basename" | cut -d. -f1)

	echo "-----------------------------------------------------"
	echo "INFO: Running tabelize.py for: $file_basename"
	# Execute tabelize.py
	# This assumes tabelize.py creates/modifies files in the CWD (which is ER_TABLES_DIR).
	"$TABELIZE_SCRIPT_PATH" -i "$f" -n "$name_without_ext" -c >> "$QUERIES_LOG_FILE"
	
	if [ $? -ne 0 ]; then
        echo "ERROR: tabelize.py failed for $f. Check $QUERIES_LOG_FILE and tabelize.py output/docs for details."
    else
        echo "INFO: Successfully processed (or tabelize.py ran without error for) $f"
    fi
done

echo "-----------------------------------------------------"
echo "INFO: Finished processing all files with tabelize.py."
echo "INFO: Standardized files (e.g., standard_drugs_atc.csv.gz) should now be in $ER_TABLES_DIR if tabelize.py was successful and configured correctly."
echo "INFO: Query log (appended): $QUERIES_LOG_FILE"

# Go back to original directory
cd "$WORKSPACE_ROOT"
echo "INFO: Returned to: $(pwd)"

exit 0 