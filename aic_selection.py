import pandas as pd
import pickle
import statsmodels.api as sm
import argparse

def perform_aic_selection(model_path, antibody_name):
    """
    Performs AIC-based forward selection on the top 10 features
    from a pre-trained RandomForest model.
    """
    # Load data
    try:
        df = pd.read_csv('cleanbackup.csv')
    except FileNotFoundError:
        print("Error: 'cleanbackup.csv' not found. Make sure the script is in the correct directory.")
        return

    # Define features and target
    target_col = f'{antibody_name} EU'
    if antibody_name == 'OmpC':
        target_col = 'OmpC. EU'
    elif antibody_name == 'Cbir1':
        target_col = 'Cbir1 EU'
    elif antibody_name == 'IgA ASCA':
        target_col = 'IgA ASCA EU'
    elif antibody_name == 'IgG ASCA':
        target_col = 'IgG ASCA EU'
        
    y = df[target_col]
    X = df.drop(['IgA ASCA EU','IgG ASCA EU','OmpC. EU','Cbir1 EU','ANCA EU','serum_id', 'participant_id', 'sample_name'], axis=1)

    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            rf_model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Get feature importance from the trained model
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame with feature names and their importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Sort by importance in descending order
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # Get the top 10 features from the Random Forest model
    top_10_features = feature_importance_df.head(10)['Feature'].tolist()

    # Prepare the dataframe with only top features
    X_top_features = X[top_10_features]

    aic_results = []

    print(f"--- AIC-based Forward Selection for {antibody_name} ---")
    print('='*60)

    # Iteratively add features and calculate AIC
    for i in range(1, len(top_10_features) + 1):
        selected_features = top_10_features[:i]
        X_subset = sm.add_constant(X_top_features[selected_features])  # Add constant for intercept
        
        model = sm.OLS(y, X_subset).fit()
        
        aic_results.append({
            'num_features': i,
            'features': ", ".join(selected_features),
            'AIC': model.aic
        })

    # Create a DataFrame for the results
    aic_df = pd.DataFrame(aic_results)

    # Find the model with the lowest AIC
    best_model_aic = aic_df.loc[aic_df['AIC'].idxmin()]

    print('AIC Comparison Summary:')
    print(aic_df[['num_features', 'AIC']].to_string(index=False))
    print('\\n' + '='*60)
    print(f"Best model has {best_model_aic['num_features']} features with an AIC of {best_model_aic['AIC']:.2f}")
    print(f"Features in best model: {best_model_aic['features']}")
    print('='*60 + '\\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform AIC selection for a given model.')
    parser.add_argument('model', help='Path to the .pkl model file (e.g., ANCAmodel.pkl)')
    parser.add_argument('antibody', help='Name of the antibody (e.g., ANCA)')
    
    args = parser.parse_args()
    
    perform_aic_selection(args.model, args.antibody) 