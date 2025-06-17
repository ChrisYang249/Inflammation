import pandas as pd

# Paths to your files
metadata_file = '../hmp2_metagenomics_metadata.xlsx'  # Update this if the path is different
unique_ids_file = '../unique_participant_ids.csv'
output_file = '../filtered_metagenomics_metadata.csv'
output_file2 = '../non-duplicate_metagenomics_metadata.csv'

# Read the files
metadata_df = pd.read_excel(metadata_file)
unique_ids_df = pd.read_csv(unique_ids_file)

# Get the set of unique participant IDs
unique_ids = set(unique_ids_df['Participant ID'].astype(str))

# Filter metadata for rows with participant IDs in the unique list
filtered_df = metadata_df[metadata_df['Participant ID'].astype(str).isin(unique_ids)]

# Select only the columns of interest
filtered_df = filtered_df[['Participant ID', 'External ID', 'data_type']]

# Save to new CSV
filtered_df.to_csv(output_file, index=False) 

# Remove duplicate participant IDs, keeping only the first occurrence
filtered_df2= filtered_df.drop_duplicates(subset=['Participant ID'], keep='first')

# Save to new CSV
filtered_df2.to_csv(output_file2, index=False) 