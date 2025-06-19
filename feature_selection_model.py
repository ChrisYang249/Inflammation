import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

# Read the data
print("Loading data...")
df = pd.read_csv('cleanbackup.csv')

# Remove duplicates keeping first occurrence
print("Removing duplicates...")
df = df.drop_duplicates(subset=['participant_id'], keep='first')

# Define the top features based on importance scores
top_features = [
    'age',
    'Phocaeicola_vulgatus',
    'Parasutterella_excrementihominis',
    'Sutterella_wadsworthensis',
    'Parabacteroides_distasonis',
    'Phocaeicola_plebeius',
    'Phocaeicola_dorei',
    'Bacteroides_caccae',
    'Bacteroides_ovatus',
    'Dialister_invisus'
]

# Prepare the data
X = df[top_features]
y = df['IgA ASCA EU']

print(f"\nDataset shape after feature selection:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model with some tuned parameters
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=3,  # Increased to reduce overfitting
    min_samples_split=6,  # Increased to reduce overfitting
    max_depth=10,        # Limited depth to prevent overfitting
    random_state=42
)

# Fit the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics on test set
print("\nTest Set Metrics:")
print("=" * 50)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Explained Variance: {explained_variance_score(y_test, y_pred):.4f}")

# Perform cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross-validation Results:")
print("=" * 50)
print("R2 scores for each fold:", cv_scores)
print(f"Mean R2: {np.mean(cv_scores):.4f}")
print(f"Std R2: {np.std(cv_scores):.4f}")

# Feature importance for selected features
feature_importance = pd.DataFrame({
    'Feature': top_features,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance for Selected Features:")
print("=" * 50)
for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']:<35} {row['Importance']:.4f}")

# Save feature importance plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance)), feature_importance['Importance'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Feature Importance Score')
plt.title('Feature Importance Scores for Selected Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('selected_features_importance.png')
print("\nFeature importance plot saved as 'selected_features_importance.png'") 