import pandas as pd
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
samples_file = os.path.join(base_dir, 'samples_with_participant_id.csv')

# Load the samples file
df_samples = pd.read_csv(samples_file)

# Remove rows where either 'Serum #id' or 'Participant ID' is blank or missing
df_samples = df_samples.dropna(subset=['Serum #id', 'Participant ID'])

# Overwrite the original file
df_samples.to_csv(samples_file, index=False)

print("Rows with missing 'Serum #id' or 'Participant ID' have been removed and samples_with_participant_id.csv updated.")