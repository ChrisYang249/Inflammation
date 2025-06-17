import pandas as pd

# Load files
backup = pd.read_csv('/Users/christopheryang/Desktop/clean data for analysis/backup.csv')
elisa = pd.read_csv('/Users/christopheryang/Desktop/clean data for analysis/serology_elisa.csv')

# Remove rows with missing serum_id
backup = backup[backup['serum_id'].notna() & (backup['serum_id'] != '')]
# Save to new CSV
backup.to_csv('/Users/christopheryang/Desktop/clean data for analysis/cleanbackup.csv', index=False)