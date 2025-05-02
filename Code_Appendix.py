# SELECTED CODE FROM FINAL.py
# This appendix includes key functions for:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
    print("Warning: imblearn not found. SMOTE will not be used.")

plt.style.use('fivethirtyeight')
sns.set_palette("deep")


# --- Label Creation ---
def create_ballon_dor_labels(df):
    """
    Creates a binary target variable 'ballon_dor_winner' (1 if won, 0 otherwise).
    Identifies winners based on a predefined list and season.
    If no winners are identified or 'season' is missing, creates synthetic labels
    based on a simple performance score heuristic.

    Args:
        df (pd.DataFrame): The input DataFrame after basic processing.

    Returns:
        pd.DataFrame: The DataFrame with the 'ballon_dor_winner' column added.
    """
    # ... (Paste the entire content of your create_ballon_dor_labels function here) ...
    # Make sure all comments within the function are included.
    if df.empty:
        print("Input DataFrame is empty. Cannot create Ballon d'Or labels.")
        return df

    data = df.copy()

    player_col = None
    if 'player' in data.columns: player_col = 'player'
    elif 'player_name' in data.columns: player_col = 'player_name'
    else:
        possible_name_cols = [col for col in data.columns if 'name' in col.lower() or 'player' in col.lower()]
        player_col = possible_name_cols[0] if possible_name_cols else None

    if not player_col or player_col not in data.columns:
         print("Warning: Player column not found or invalid. Creating dummy player IDs.")
         data['dummy_player'] = "Player_" + data.index.astype(str)
         player_col = 'dummy_player'

    data['ballon_dor_winner'] = 0

    ballon_dor_winners = {
        '2015-2016': 'Cristiano Ronaldo', '2016-2017': 'Cristiano Ronaldo',
        '2017-2018': 'Luka Modric', '2018-2019': 'Lionel Messi',
        '2019-2020': 'Not Awarded', '2020-2021': 'Lionel Messi',
        '2021-2022': 'Karim Benzema', '2022-2023': 'Lionel Messi',
        '2023-2024': 'Vinicius Junior',
    }

    if 'season' in data.columns and player_col != 'dummy_player':
        print("Applying actual Ballon d'Or labels based on season and player names.")
        for season, winner in ballon_dor_winners.items():
            if winner != 'Not Awarded':
                season_mask = data['season'].astype(str) == season
                winner_parts = winner.lower().split()
                winner_mask = data[player_col].str.lower().apply(
                    lambda x: all(part in x for part in winner_parts)
                )
                data.loc[season_mask & winner_mask, 'ballon_dor_winner'] = 1
    else:
        print("⚠️ Warning: 'season' column not found or player column invalid. Cannot apply historical Ballon d'Or labels accurately.")

    winner_count = data['ballon_dor_winner'].sum()

    if winner_count == 0:
        print("⚠️ Warning: No actual Ballon d'Or winners identified. Creating synthetic labels based on performance heuristic.")
        data['performance_score'] = 0
        goal_cols = [col for col in data.columns if 'goal' in col.lower()]
        assist_cols = [col for col in data.columns if 'assist' in col.lower()]
        trophy_cols = [col for col in data.columns if 'trophy' in col.lower() or 'title' in col.lower()]

        if goal_cols: data['performance_score'] += data[goal_cols[0]] * 3
        if assist_cols: data['performance_score'] += data[assist_cols[0]] * 2
        if trophy_cols: data['performance_score'] += data[trophy_cols[0]] * 5

        if data['performance_score'].sum() == 0:
            numeric_cols = data.select_dtypes(include=['number']).columns
            relevant_cols = [col for col in numeric_cols if col not in ['performance_score', 'ballon_dor_winner']]
            if relevant_cols:
                for col in relevant_cols:
                    if data[col].max() > 0:
                        data['performance_score'] += data[col] / data[col].max()

        if 'season' in data.columns:
            for season in data['season'].unique():
                season_data = data[data['season'] == season]
                if not season_data.empty and season_data['performance_score'].max() > 0:
                    max_idx = season_data['performance_score'].idxmax()
                    data.loc[max_idx, 'ballon_dor_winner'] = 1
        else:
            top_n = max(1, int(len(data) * 0.05))
            top_performers = data.nlargest(top_n, 'performance_score').index
            data.loc[top_performers, 'ballon_dor_winner'] = 1

        data = data.drop('performance_score', axis=1)
        winner_count = data['ballon_dor_winner'].sum()
        print(f"Synthetic labeling complete. Identified {winner_count} synthetic winners.")

    return data

# --- Feature Engineering (KPI Calculation) ---
def calculate_kpi_rating(df):
    """
    Calculates a composite Key Performance Indicator (KPI) rating for each player.
    The rating is based on various performance metrics and team success,
    and then normalized to a 0-100 scale. Also assigns a performance category.

    Args:
        df (pd.DataFrame): The input DataFrame, ideally with processed and labeled data.

    Returns:
        pd.DataFrame: The DataFrame with 'kpi_rating' and 'kpi_category' columns added.
    """
    # ... (Paste the entire content of your calculate_kpi_rating function here) ...
    # Make sure all comments within the function are included.
    data = df.copy()

    goal_col = next((col for col in data.columns if 'goal' in col.lower()), None)
    assist_col = next((col for col in data.columns if 'assist' in col.lower()), None)
    minutes_col = next((col for col in data.columns if 'minute' in col.lower()), None)
    appearances_col = next((col for col in data.columns if 'appear' in col.lower() or 'match' in col.lower()), None)
    shot_col = next((col for col in data.columns if 'shot' in col.lower()), None)
    champions_league = next((col for col in data.columns if 'champions' in col.lower() or 'ucl' in col.lower()), None)
    league_title = next((col for col in data.columns if 'league' in col.lower() and ('title' in col.lower() or 'trophy' in col.lower())), None)
    world_cup = next((col for col in data.columns if 'world cup' in col.lower() or 'international' in col.lower()), None)
    team_col = next((col for col in data.columns if 'team' in col.lower()), None)

    data['goal_score'] = data[goal_col] * 3 if goal_col else 0
    data['assist_score'] = data[assist_col] * 2 if assist_col else 0

    if goal_col and shot_col and data[shot_col].max() > 0:
        data['efficiency_score'] = (data[goal_col] / data[shot_col].replace(0, 1)) * 15
    else: data['efficiency_score'] = 0

    data['production'] = (data[goal_col] if goal_col else 0) + (data[assist_col] if assist_col else 0)
    if 'production' in data.columns and data['production'].sum() > 0:
        if minutes_col and data[minutes_col].max() > 0:
            data['production_rate_score'] = (data['production'] / data[minutes_col].replace(0, 90)) * 90 * 10
        elif appearances_col and data[appearances_col].max() > 0:
            data['production_rate_score'] = (data['production'] / data[appearances_col].replace(0, 1)) * 10
        else: data['production_rate_score'] = 0
    else: data['production_rate_score'] = 0

    data['trophy_score'] = 0
    if champions_league: data['trophy_score'] += data[champions_league] * 25
    if world_cup: data['trophy_score'] += data[world_cup] * 30
    if league_title: data['trophy_score'] += data[league_title] * 15

    data['team_success_score'] = 0
    if team_col and champions_league:
        teams = data[team_col].unique()
        for team in teams:
            if team != 'Unknown':
                team_players_mask = data[team_col] == team
                if not data.loc[team_players_mask, champions_league].empty and data.loc[team_players_mask, champions_league].max() > 0:
                    data.loc[team_players_mask, 'team_success_score'] += 10

    data['kpi_rating'] = (
        data['goal_score'] + data['assist_score'] + data['efficiency_score'] +
        data['production_rate_score'] + data['trophy_score'] + data['team_success_score']
    )

    if data['kpi_rating'].max() > 0:
        data['kpi_rating'] = (data['kpi_rating'] / data['kpi_rating'].max()) * 100
    else: print("Warning: Max KPI rating is 0. Cannot normalize.")

    data['kpi_rating'] = data['kpi_rating'].round(1)

    conditions = [
        (data['kpi_rating'] >= 75), (data['kpi_rating'] >= 65) & (data['kpi_rating'] < 75),
        (data['kpi_rating'] >= 55) & (data['kpi_rating'] < 65), (data['kpi_rating'] >= 45) & (data['kpi_rating'] < 55),
        (data['kpi_rating'] >= 35) & (data['kpi_rating'] < 45), (data['kpi_rating'] < 35)
    ]
    values = ['World Class', 'Elite', 'Excellent', 'Very Good', 'Good', 'Average']
    data['kpi_category'] = np.select(conditions, values, default='Average')
    print("Calculated KPI ratings and categories.")

    return data

# Note: The visualization function (visualize_kpi_data) is omitted here for brevity.

# --- Model Building ---
def build_prediction_model(df):
    """
    Builds and evaluates machine learning models (Random Forest, Logistic Regression, SVM)
    to predict Ballon d'Or winners based on player features.
    Handles class imbalance using SMOTE if available.
    Visualizes feature importance (from Random Forest) and ROC curves for model comparison.

    Args:
        df (pd.DataFrame): The input DataFrame with features and the 'ballon_dor_winner' target.

    Returns:
        tuple: A tuple containing:
            - best_model (Pipeline): The best performing model pipeline (based on accuracy).
            - feature_cols (list): List of names of the features used for training.
            - importance_df (pd.DataFrame): DataFrame of feature importances (from RF), or empty.
            - all_models (dict): Dictionary of all trained model pipelines.
    """
    # ... (Paste the entire content of your build_prediction_model function here) ...
    # Make sure all comments within the function are included.
    if df.empty or 'ballon_dor_winner' not in df.columns:
        print("Error: Input DataFrame is empty or missing 'ballon_dor_winner' column for model building.")
        return None, [], pd.DataFrame(), {}

    exclude_cols = ['ballon_dor_winner', 'index', 'id', 'data_source', 'season', 'kpi_rating', 'kpi_category']
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(object_cols)
    exclude_cols = list(set(exclude_cols))

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not feature_cols:
        print("Error: No numerical feature columns found after exclusion.")
        return None, [], pd.DataFrame(), {}

    print(f"Using {len(feature_cols)} features for model building.")

    X = df[feature_cols].copy()
    y = df['ballon_dor_winner'].copy()
    X.fillna(0, inplace=True)

    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        print("Error: Target variable 'ballon_dor_winner' has only one class. Cannot train a classifier.")
        return None, feature_cols, pd.DataFrame(), {}
    print(f"Class distribution - Non-winners (0): {class_counts[0]}, Winners (1): {class_counts[1]}")
    if class_counts[1] < 5: print("⚠️ Warning: Very few positive samples (Ballon d'Or winners) in dataset. Model may be unreliable or heavily biased.")

    if SMOTE is not None:
        try:
            print("Attempting to resample data using SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"Data resampled using SMOTE. New distribution: {np.bincount(y_resampled)}")
            X, y = X_resampled, y_resampled
        except Exception as e:
            print(f"Error during SMOTE resampling: {e}. Using original imbalanced data.")
    else:
        print("SMOTE (from imblearn) not available. Using original imbalanced data.")

    print("Splitting data into training (80%) and testing (20%) sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    print("Setting up model pipelines (Random Forest, Logistic Regression, SVM).")
    rf_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42))])
    lr_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42))])
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42))])

    models = {'Random Forest': rf_pipeline, 'Logistic Regression': lr_pipeline, 'SVM': svm_pipeline}

    model_metrics = {}; model_predictions = {}; model_probas = {}

    try:
        for name, pipeline in models.items():
            print(f"\nTraining {name} model...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            model_metrics[name] = {'accuracy': accuracy, 'conf_matrix': conf_matrix}
            model_predictions[name] = y_pred
            model_probas[name] = y_pred_proba
            print(f"{name} Model Accuracy: {accuracy:.4f}")
            print(f"{name} Confusion Matrix:\n{conf_matrix}")

        importance_df = pd.DataFrame()
        if 'Random Forest' in models:
            try:
                rf_model = models['Random Forest'].named_steps['classifier']
                feature_importance = rf_model.feature_importances_
                importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importance}).sort_values('Importance', ascending=False)
                plt.figure(figsize=(12, 6)); sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis'); plt.title('Top 10 Feature Importance for Ballon d\'Or Prediction (Random Forest)', fontsize=14); plt.xlabel('Importance', fontsize=12); plt.ylabel('Feature', fontsize=12); plt.tight_layout(); plt.show(); print("Visualized top 10 feature importances.")
            except Exception as e: print(f"Could not get or plot feature importance for Random Forest: {e}")

        if len(np.unique(y_test)) > 1:
            print("\nGenerating ROC Curves for model comparison...")
            plt.figure(figsize=(10, 8))
            for name, y_pred_proba in model_probas.items():
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                color = 'darkorange' if name == 'Random Forest' else ('blue' if name == 'Logistic Regression' else 'green')
                linestyle = '-' if name == 'Random Forest' else ('--' if name == 'Logistic Regression' else '-.')
                plt.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle, label=f'{name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Guess (AUC = 0.50)')
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12); plt.title('ROC Curves for Different Models', fontsize=14); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.show(); print("Visualized ROC curves.")
            print("\n--- Model Performance Summary ---");
            for name, metrics in model_metrics.items(): print(f"{name}: Accuracy = {metrics['accuracy']:.4f}")
            print("-----------------------------")
        else: print("Skipping ROC curve visualization: Only one class present in the test set.")

        if model_metrics:
            best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['accuracy'])
            best_model = models[best_model_name]
            print(f"\nSelected best model based on accuracy: {best_model_name} (Accuracy: {model_metrics[best_model_name]['accuracy']:.4f})")
        else: best_model_name = "None"; best_model = None; print("\nNo models were successfully trained or evaluated.")

        return best_model, feature_cols, importance_df, models

    except Exception as e:
        print(f"An error occurred during model building or evaluation: {str(e)}")
        return None, feature_cols, pd.DataFrame(), {}

# --- Prediction ---
def predict_ballon_dor_candidates(df, model, feature_cols, year=2025, all_models=None):
    """
    Uses the trained model(s) to predict the probability of players winning the Ballon d'Or.
    Predicts based on the latest season's data if available, otherwise uses all data.
    Combines model probability with KPI rating (if available) for a final score.
    Ranks candidates and visualizes the top candidates with their probabilities and KPIs.
    Also provides a comparison visualization of predictions across different models for top players.

    Args:
        df (pd.DataFrame): The DataFrame containing player data, including features and KPIs.
        model (Pipeline): The primary (best) trained model pipeline.
        feature_cols (list): List of feature column names used by the model.
        year (int): The year for which the prediction is being made (for visualization titles). Default is 2025.
        all_models (dict, optional): Dictionary containing all trained model pipelines for comparison.

    Returns:
        pd.DataFrame: A DataFrame of predicted candidates sorted by probability, including rank.
    """
    # ... (Paste the entire content of your predict_ballon_dor_candidates function here) ...
    # Make sure all comments within the function are included.
    if model is None or not feature_cols or df.empty:
        print("Error: Model not trained, no features available, or input data is empty. Cannot make predictions.")
        return pd.DataFrame()

    player_col = None
    if 'player' in df.columns: player_col = 'player'
    elif 'player_name' in df.columns: player_col = 'player_name'
    else:
        possible_name_cols = [col for col in df.columns if 'name' in col.lower() or 'player' in col.lower()]
        player_col = possible_name_cols[0] if possible_name_cols else None

    if not player_col or player_col not in df.columns:
        print("Error: Could not identify a valid player name column for predictions.")
        df['dummy_player'] = "Player_" + df.index.astype(str)
        player_col = 'dummy_player'
        print(f"Using dummy player column '{player_col}' for predictions.")

    if 'season' in df.columns and df['season'].nunique() > 0:
        seasons = sorted(df['season'].astype(str).unique())
        latest_season = seasons[-1]
        prediction_data = df[df['season'].astype(str) == latest_season].copy()
        print(f"Making predictions based on data from the latest season: {latest_season}")
    else:
        prediction_data = df.copy()
        print("Making predictions based on all available data (no 'season' column found).")

    missing_features = [col for col in feature_cols if col not in prediction_data.columns]
    if missing_features:
        print(f"Adding {len(missing_features)} missing feature columns with value 0 to prediction data: {missing_features}")
        for col in missing_features:
            prediction_data[col] = 0

    X_pred = prediction_data[feature_cols].copy()
    X_pred.fillna(0, inplace=True)

    try:
        prediction_data['predicted_probability'] = model.predict_proba(X_pred)[:, 1]
        print(f"Predictions made using the primary model.")
    except Exception as e:
        print(f"Error during prediction with primary model: {e}")
        prediction_data['predicted_probability'] = 0

    if all_models:
        print("Getting predictions from all trained models for comparison.")
        try:
            for name, pipeline in all_models.items():
                model_col_name = f'probability_{name.lower().replace(" ", "_")}'
                prediction_data[model_col_name] = pipeline.predict_proba(X_pred)[:, 1]
                print(f"Got predictions from {name} model.")
        except Exception as e:
            print(f"Error getting predictions from comparison models: {e}")

    if 'kpi_rating' in prediction_data.columns:
        print("Blending model probability with KPI rating.")
        max_kpi = prediction_data['kpi_rating'].max()
        if max_kpi > 0:
            normalized_kpi = prediction_data['kpi_rating'] / max_kpi
            prediction_data['predicted_probability'] = (
                0.6 * prediction_data['predicted_probability'] +
                0.4 * normalized_kpi
            ).clip(0, 1) # Ensure bounds
        else: print("Warning: Max KPI rating is 0. Cannot blend KPI into prediction probability.")
    else: print("KPI rating column not found. Prediction probability is solely model-based.")

    candidates = prediction_data[[player_col, 'predicted_probability']].copy()
    if all_models:
        for name in all_models.keys():
            model_col = f'probability_{name.lower().replace(" ", "_")}'
            if model_col in prediction_data.columns: candidates[model_col] = prediction_data[model_col]

    if 'kpi_rating' in prediction_data.columns: candidates['kpi_rating'] = prediction_data['kpi_rating']
    if 'kpi_category' in prediction_data.columns: candidates['kpi_category'] = prediction_data['kpi_category']

    candidates = candidates.sort_values('predicted_probability', ascending=False).reset_index(drop=True)
    candidates['rank'] = range(1, len(candidates) + 1)

    if all_models and len(all_models) > 1:
        print("\nGenerating model prediction comparison plot for top candidates...")
        top_n_compare = min(10, len(candidates))
        if top_n_compare > 0:
             top_candidates_compare = candidates.head(top_n_compare).copy()
             model_cols_to_plot = ['predicted_probability'] + [f'probability_{name.lower().replace(" ", "_")}' for name in all_models.keys() if f'probability_{name.lower().replace(" ", "_")}' in candidates.columns]
             model_names_for_plot = ['Combined Score'] + list(all_models.keys())
             valid_model_cols = [col for col in model_cols_to_plot if col in top_candidates_compare.columns]
             valid_model_names = [model_names_for_plot[model_cols_to_plot.index(col)] for col in valid_model_cols]
             if len(valid_model_cols) > 1:
                 x = np.arange(len(top_candidates_compare)); width = 0.8 / len(valid_model_cols); fig, ax = plt.subplots(figsize=(16, 8))
                 for i, (col, name) in enumerate(zip(valid_model_cols, valid_model_names)):
                     offset = width * i - width * len(valid_model_cols) / 2 + width / 2
                     bars = ax.bar(x + offset, top_candidates_compare[col], width, label=name)
                     for bar in bars: height = bar.get_height(); ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                 ax.set_xlabel('Player', fontsize=12); ax.set_ylabel('Probability', fontsize=12); ax.set_title(f'Model Comparison: Top {top_n_compare} Predicted Ballon d\'Or {year} Candidates', fontsize=16); ax.set_xticks(x); ax.set_xticklabels(top_candidates_compare[player_col], rotation=45, ha='right'); ax.legend(); ax.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.show(); print("Generated model comparison plot.")
             else: print("Warning: Not enough valid probability columns to generate model comparison plot.")
        else: print("Warning: Not enough candidates to generate model comparison plot.")

    print(f"\nGenerating plot for top 20 predicted Ballon d'Or {year} candidates...")
    plt.figure(figsize=(16, 8))
    top20 = candidates.head(20)
    if not top20.empty:
        if 'kpi_rating' in top20.columns: colors = plt.cm.viridis(top20['kpi_rating'] / 100); bars = plt.bar(top20[player_col], top20['predicted_probability'], color=colors); print("Using KPI rating to color bars.")
        else: bars = plt.bar(top20[player_col], top20['predicted_probability'], color='lightblue'); print("KPI rating not available for coloring bars.")
        for i in range(min(3, len(bars))): bars[i].set_color(['gold', 'silver', '#cd7f32'][i])
        for i, bar in enumerate(bars):
            height = bar.get_height(); plt.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
            if 'kpi_rating' in top20.columns: kpi = top20.iloc[i]['kpi_rating']; plt.annotate(f'KPI: {kpi}', xy=(bar.get_x() + bar.get_width() / 2, height/2), ha='center', fontsize=9, color='white')
        plt.xlabel('Player', fontsize=12); plt.ylabel('Predicted Probability of Winning', fontsize=12); plt.title(f'Top {len(top20)} Predicted Ballon d\'Or {year} Candidates with KPI Ratings', fontsize=16); plt.xticks(rotation=45, ha='right'); plt.ylim(0, min(1, top20['predicted_probability'].max() * 1.2)); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.show(); print(f"Generated plot for top {len(top20)} candidates.")
    else: print("Warning: No candidates found with predicted probabilities to visualize.")

    print(f"\n--- Top 10 Predicted Ballon d'Or {year} Candidates ---")
    if not candidates.empty:
        for i, (_, row) in enumerate(candidates.head(10).iterrows()):
            kpi_info = f" - KPI: {row['kpi_rating']} ({row['kpi_category']})" if 'kpi_rating' in row else ""
            print(f"{i+1}. {row[player_col]} - {row['predicted_probability']:.1%}{kpi_info}")
    else: print("No candidates found.")
    print("----------------------------------------------------")

    return candidates

# Note: The main execution block (__main__ part) is omitted here for brevity,
# but it orchestrates the calls to the functions included above.