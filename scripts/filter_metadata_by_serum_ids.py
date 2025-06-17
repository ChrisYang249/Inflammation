import pandas as pd
import re

# Path to serology file and metadata CSV
serology_file = '../hmp2_serology_Compiled_ELISA_Data.tsv'
metadata_csv = '../hmp2_metadata_2018-08-20.csv'

# Read the serology file (assuming first row is header, columns B onward are serum IDs)
serology_df = pd.read_csv(serology_file, sep='\t')
serum_ids = set(serology_df.columns[1:])  # Skip the first column (row labels)

# Read the metadata CSV
metadata_df = pd.read_csv(metadata_csv)

# Function to extract numeric part before any non-digit character
def extract_base_id(val):
    match = re.match(r"(\d+)", str(val))
    return match.group(1) if match else None

# Filter rows where any serum #1-4 matches a serum ID from the serology file
serum_cols = ['Serum #1', 'Serum #2', 'Serum #3', 'Serum #4']
mask = metadata_df[serum_cols].apply(
    lambda row: any(extract_base_id(val) in serum_ids for val in row if extract_base_id(val)), axis=1
)
filtered_df = metadata_df.loc[mask, serum_cols + ['External ID', 'Participant ID', 'data_type']]

# After filtering, clean up the serum columns to keep only the numeric part
for col in serum_cols:
    filtered_df[col] = filtered_df[col].apply(lambda val: extract_base_id(val) if extract_base_id(val) in serum_ids else val)

# Remove duplicate rows based on the serum columns
filtered_df = filtered_df.drop_duplicates(subset=serum_cols)

# Save or use filtered_df as needed
filtered_df.to_csv('../filtered_metadata_by_serum_ids.csv', index=False)

# Create a new CSV with unique Participant IDs only
unique_participants = filtered_df[['Participant ID']].drop_duplicates()
unique_participants.to_csv('../unique_participant_ids.csv', index=False) 