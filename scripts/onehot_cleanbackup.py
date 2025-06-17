import pandas as pd

# One-hot encoding mapping for reference:
# sex:
#   'Male'   -> sex_Male (1 if Male, 0 otherwise)
#   'Female' -> sex_Female (1 if Female, 0 otherwise)
# race:
#   'White' -> race_White (1 if White, 0 otherwise)
#   'Black or African American' -> race_Black or African American (1 if Black or African American, 0 otherwise)
#   'More than one race' -> race_More than one race (1 if More than one race, 0 otherwise)
#   'Other' -> race_Other (1 if Other, 0 otherwise)

# Load the cleaned backup file
df = pd.read_csv('/Users/christopheryang/Desktop/clean data for analysis/cleanbackup.csv')

# One-hot encode 'sex' and 'race'
df_encoded = pd.get_dummies(df, columns=['sex', 'race'])

# Convert boolean columns to int (1/0)
for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype(int)

# Save the result
df_encoded.to_csv('/Users/christopheryang/Desktop/clean data for analysis/cleanbackup.csv', index=False)