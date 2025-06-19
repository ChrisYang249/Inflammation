import pandas as pd

# Read the CSV file
print("Reading the CSV file...")
df = pd.read_csv('backup2.csv')

# Print initial information
print(f"\nInitial number of rows: {len(df)}")
print(f"Number of unique participant_ids: {df['participant_id'].nunique()}")
print(f"Number of duplicate participant_ids: {len(df) - df['participant_id'].nunique()}")

# Keep only the first occurrence of each participant_id
df_unique = df.drop_duplicates(subset=['participant_id'], keep='first')

# Print results
print(f"\nFinal number of rows after removing duplicates: {len(df_unique)}")
print(f"Number of rows removed: {len(df) - len(df_unique)}")

# Save the deduplicated data to a new CSV file
output_file = 'backup2_no_duplicates.csv'
df_unique.to_csv(output_file, index=False)
print(f"\nSaved deduplicated data to: {output_file}")

# Print sample of removed duplicates for verification
print("\nExample of removed duplicate entries:")
duplicates = df[df['participant_id'].duplicated(keep='first')]
print(duplicates.head().to_string()) 