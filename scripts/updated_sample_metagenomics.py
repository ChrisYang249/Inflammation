import pandas as pd

# Load the sample names from the TSV file
samples_df = pd.read_csv('../selected_samples_hmp2.tsv', sep='\t', usecols=['sample_name'])
sample_names = set(samples_df['sample_name'])

# Load the metadata from the CSV file
metadata_df = pd.read_csv('../hmp2_metadata_2018-08-20.csv')

# Filter metadata to only include rows with External ID in sample_names
filtered_metadata = metadata_df[metadata_df['External ID'].isin(sample_names)]

# Keep only the specified columns
columns_to_keep = ['External ID', 'Participant ID', 'site_sub_coll']
filtered_metadata = filtered_metadata[columns_to_keep]

# Save the filtered DataFrame to a CSV file
filtered_metadata.to_csv('../updated_sample_metagenomics.csv', index=False)