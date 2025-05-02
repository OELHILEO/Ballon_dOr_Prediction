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
import sys

plt.style.use('fivethirtyeight')
sns.set_palette("deep")

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep=";")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def process_player_data(df):
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.copy()
        
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        if 'player' in df.columns:
            player_col = 'player'
        elif 'player_name' in df.columns:
            player_col = 'player_name'
        else:
            possible_name_cols = [col for col in df.columns if 'name' in col.lower() or 'player' in col.lower()]
            player_col = possible_name_cols[0] if possible_name_cols else None
        
        if player_col:
            df[player_col] = df[player_col].str.strip().str.title()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        return df
    return pd.DataFrame()

def create_ballon_dor_labels(df):
    if df.empty:
        return df

    data = df.copy()

    if 'player' in data.columns:
        player_col = 'player'
    elif 'player_name' in data.columns:
        player_col = 'player_name'
    else:
        possible_name_cols = [col for col in data.columns if 'name' in col.lower() or 'player' in col.lower()]
        player_col = possible_name_cols[0] if possible_name_cols else None

    if not player_col:
        data['dummy_player'] = "Player_" + data.index.astype(str)
        player_col = 'dummy_player'

    data['ballon_dor_winner'] = 0

    ballon_dor_winners = {
        '2015-2016': 'Cristiano Ronaldo',
        '2016-2017': 'Cristiano Ronaldo',
        '2017-2018': 'Luka Modric',
        '2018-2019': 'Lionel Messi',
        '2019-2020': 'Not Awarded', 
        '2020-2021': 'Lionel Messi',
        '2021-2022': 'Karim Benzema',
        '2022-2023': 'Lionel Messi',
        '2023-2024': 'Vinicius Junior',
    }

    if 'season' in data.columns:
        for season, winner in ballon_dor_winners.items():
            if winner != 'Not Awarded':
                season_mask = data['season'] == season
                winner_parts = winner.lower().split()
                winner_mask = data[player_col].str.lower().apply(
                    lambda x: all(part in x for part in winner_parts)
                )
                data.loc[season_mask & winner_mask, 'ballon_dor_winner'] = 1
    else:
        print("⚠️ Warning: 'season' column not found. Ballon d'Or winners cannot be properly labeled.")

    winner_count = data['ballon_dor_winner'].sum()
    
    if winner_count == 0:
        print("⚠️ Warning: No Ballon d'Or winners identified in the dataset. Creating synthetic labels based on performance.")
        
        data['performance_score'] = 0
        
        goal_cols = [col for col in data.columns if 'goal' in col.lower()]
        assist_cols = [col for col in data.columns if 'assist' in col.lower()]
        trophy_cols = [col for col in data.columns if 'trophy' in col.lower() or 'title' in col.lower()]
        
        if goal_cols:
            data['performance_score'] += data[goal_cols[0]] * 3
        
        if assist_cols:
            data['performance_score'] += data[assist_cols[0]] * 2
            
        if trophy_cols:
            data['performance_score'] += data[trophy_cols[0]] * 5
            
        if data['performance_score'].sum() == 0:
            numeric_cols = data.select_dtypes(include=['number']).columns
            relevant_cols = [col for col in numeric_cols if col != 'performance_score' and col != 'ballon_dor_winner']
            
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

    return data

def calculate_kpi_rating(df):
    data = df.copy()
    
    goal_col = next((col for col in data.columns if 'goal' in col.lower()), None)
    assist_col = next((col for col in data.columns if 'assist' in col.lower()), None)
    minutes_col = next((col for col in data.columns if 'minute' in col.lower()), None)
    appearances_col = next((col for col in data.columns if 'appear' in col.lower() or 'match' in col.lower()), None)
    shot_col = next((col for col in data.columns if 'shot' in col.lower()), None)
    champions_league = next((col for col in data.columns if 'champions' in col.lower() or 'ucl' in col.lower()), None)
    league_title = next((col for col in data.columns if 'league' in col.lower() and ('title' in col.lower() or 'trophy' in col.lower())), None)
    world_cup = next((col for col in data.columns if 'world cup' in col.lower() or 'international' in col.lower()), None)
    
    if goal_col:
        data['goal_score'] = data[goal_col] * 3
    else:
        data['goal_score'] = 0
        
    if assist_col:
        data['assist_score'] = data[assist_col] * 2
    else:
        data['assist_score'] = 0
    
    if goal_col and shot_col and data[shot_col].max() > 0:
        data['efficiency_score'] = (data[goal_col] / data[shot_col].replace(0, 1)) * 15
    else:
        data['efficiency_score'] = 0
    
    if goal_col and assist_col:
        data['production'] = data[goal_col] + data[assist_col]
        
        if minutes_col and data[minutes_col].max() > 0:
            data['production_rate_score'] = (data['production'] / data[minutes_col].replace(0, 90)) * 90 * 10
        elif appearances_col and data[appearances_col].max() > 0:
            data['production_rate_score'] = (data['production'] / data[appearances_col].replace(0, 1)) * 10
        else:
            data['production_rate_score'] = 0
    else:
        data['production_rate_score'] = 0
    
    data['trophy_score'] = 0
    
    if champions_league:
        data['trophy_score'] += data[champions_league] * 25
    
    if world_cup:
        data['trophy_score'] += data[world_cup] * 30
    
    if league_title:
        data['trophy_score'] += data[league_title] * 15
        
    data['team_success_score'] = 0
    team_col = next((col for col in data.columns if 'team' in col.lower()), None)
    
    if team_col and champions_league:
        teams = data[team_col].unique()
        for team in teams:
            if team != 'Unknown':
                team_players = data[data[team_col] == team].index
                if len(team_players) > 0:
                    if data.loc[team_players, champions_league].max() > 0:
                        data.loc[team_players, 'team_success_score'] += 10
    
    data['kpi_rating'] = (
        data['goal_score'] + 
        data['assist_score'] + 
        data['efficiency_score'] + 
        data['production_rate_score'] + 
        data['trophy_score'] + 
        data['team_success_score']
    )
    
    if data['kpi_rating'].max() > 0:
        data['kpi_rating'] = (data['kpi_rating'] / data['kpi_rating'].max()) * 100
    
    data['kpi_rating'] = data['kpi_rating'].round(1)
    
    conditions = [
    (data['kpi_rating'] >= 75),  
    (data['kpi_rating'] >= 65) & (data['kpi_rating'] < 75),  
    (data['kpi_rating'] >= 55) & (data['kpi_rating'] < 65),  
    (data['kpi_rating'] >= 45) & (data['kpi_rating'] < 55),  
    (data['kpi_rating'] >= 35) & (data['kpi_rating'] < 45),  
    (data['kpi_rating'] < 35)  
]
    
    values = ['World Class', 'Elite', 'Excellent', 'Very Good', 'Good', 'Average']
    data['kpi_category'] = np.select(conditions, values, default='Average')
    
    return data

def visualize_kpi_data(df):
    if df.empty:
        return None
    
    if 'player' in df.columns:
        player_col = 'player'
    elif 'player_name' in df.columns:
        player_col = 'player_name'
    else:
        possible_name_cols = [col for col in df.columns if 'name' in col.lower() or 'player' in col.lower()]
        player_col = possible_name_cols[0] if possible_name_cols else df.columns[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    top_players = df.nlargest(15, 'kpi_rating')
    
    bars = sns.barplot(x='kpi_rating', y=player_col, data=top_players, ax=axes[0], palette='viridis')
    
    for i, p in enumerate(bars.patches):
        if top_players.iloc[i]['ballon_dor_winner'] == 1:
            p.set_facecolor('gold')
    
    axes[0].set_title('Top 15 Players by KPI Rating', fontsize=14)
    axes[0].set_xlabel('KPI Rating (0-100)', fontsize=12)
    axes[0].set_ylabel('')
    
    for i, v in enumerate(top_players['kpi_rating']):
        category = top_players.iloc[i]['kpi_category']
        axes[0].text(v + 1, i, f"{category}", va='center')
    
    # Define the values variable here before using it
    values = ['World Class', 'Elite', 'Excellent', 'Very Good', 'Good', 'Average']
    
    category_counts = df['kpi_category'].value_counts().sort_index(
        key=lambda x: pd.Categorical(x, categories=values[::-1], ordered=True)
    )
    
    category_colors = {
        'World Class': '#1a9641',
        'Elite': '#a6d96a',
        'Excellent': '#ffffbf',
        'Very Good': '#fdae61',
        'Good': '#d7191c',
        'Average': '#808080'
    }
    
    colors = [category_colors.get(cat, '#808080') for cat in category_counts.index]
    
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[1], palette=colors)
    axes[1].set_title('Distribution of Players by KPI Category', fontsize=14)
    axes[1].set_xlabel('KPI Category', fontsize=12)
    axes[1].set_ylabel('Number of Players', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(category_counts.values):
        axes[1].text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def build_prediction_model(df):
    if df.empty or 'ballon_dor_winner' not in df.columns:
        return None, [], pd.DataFrame()
        
    exclude_cols = ['ballon_dor_winner', 'index', 'id', 'data_source', 'season', 'kpi_rating', 'kpi_category']
    
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(object_cols)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['ballon_dor_winner'].copy()
    
    X.fillna(0, inplace=True)
    
    class_counts = np.bincount(y)
    print(f"Class distribution - Non-winners: {class_counts[0]}, Winners: {class_counts[1]}")
    
    if class_counts[1] < 3:
        print("⚠️ Warning: Very few Ballon d'Or winners in dataset. Model may have limited predictive power.")
    
    try:
        from imblearn.over_sampling import SMOTE
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
        print(f"Data resampled using SMOTE: {np.bincount(y_resampled)}")
        X, y = X_resampled, y_resampled
    except ImportError:
        print("SMOTE not available. Using original imbalanced data.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3 different models
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    #Logistic Regression model
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        ))
    ])
    
    #SVM model
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Dictionary 
    models = {
        'Random Forest': rf_pipeline,
        'Logistic Regression': lr_pipeline,
        'SVM': svm_pipeline
    }
    
    # Evaluate all models
    model_metrics = {}
    model_predictions = {}
    model_probas = {}
    
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
        
        # Get feature importance 
        rf_model = models['Random Forest'].named_steps['classifier']
        feature_importance = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
        plt.title('Feature Importance for Ballon d\'Or Prediction (Random Forest)')
        plt.tight_layout()
        plt.show()
        
        # Compare models using ROC curves
        if len(np.unique(y_test)) > 1:
            plt.figure(figsize=(10, 8))
            
            
            for name, y_pred_proba in model_probas.items():
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
              
                if name == 'Random Forest':
                    color = 'darkorange'
                    linestyle = '-'
                elif name == 'Logistic Regression':
                    color = 'blue'
                    linestyle = '--'
                else:  # SVM
                    color = 'green'
                    linestyle = '-.'
                
                plt.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle,
                        label=f'{name} (AUC = {roc_auc:.2f})')
            
            #reference 
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves for Different Models')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print("\nModel Comparison:")
            for name, metrics in model_metrics.items():
                print(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
         # Select the best model 
        best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['accuracy'])
        best_model = models[best_model_name]
        print(f"\nBest model: {best_model_name} with accuracy {model_metrics[best_model_name]['accuracy']:.4f}")
        
        return best_model, feature_cols, importance_df, models
    
    except Exception as e:
        print(f"Error building models: {str(e)}")
        return None, feature_cols, pd.DataFrame(), {}

def predict_ballon_dor_candidates(df, model, feature_cols, year=2025, all_models=None):
    if model is None or not feature_cols or df.empty:
        return pd.DataFrame()
    
    if 'player' in df.columns:
        player_col = 'player'
    elif 'player_name' in df.columns:
        player_col = 'player_name'
    else:
        possible_name_cols = [col for col in df.columns if 'name' in col.lower() or 'player' in col.lower()]
        player_col = possible_name_cols[0] if possible_name_cols else df.columns[0]
    
    if 'season' in df.columns:
        seasons = df['season'].unique()
        latest_season = sorted(seasons)[-1]
        prediction_data = df[df['season'] == latest_season].copy()
        print(f"Making predictions based on data from season: {latest_season}")
    else:
        prediction_data = df.copy()
        print("Making predictions based on all available data")
    
    missing_features = [col for col in feature_cols if col not in prediction_data.columns]
    if missing_features:
        for col in missing_features:
            prediction_data[col] = 0
    
    X_pred = prediction_data[feature_cols].copy()
    X_pred.fillna(0, inplace=True)
    
    # Main model predictions (best model)
    prediction_data['predicted_probability'] = model.predict_proba(X_pred)[:, 1]
    
    if all_models:
        for name, pipeline in all_models.items():
            prediction_data[f'probability_{name.lower().replace(" ", "_")}'] = pipeline.predict_proba(X_pred)[:, 1]
    
    if 'kpi_rating' in prediction_data.columns:
        max_kpi = prediction_data['kpi_rating'].max()
        if max_kpi > 0:
            normalized_kpi = prediction_data['kpi_rating'] / max_kpi
            prediction_data['predicted_probability'] = (
                0.6 * prediction_data['predicted_probability'] + 
                0.4 * normalized_kpi
            )
    
    candidates = prediction_data[[player_col, 'predicted_probability']].copy()
    
    if all_models:
        for name in all_models.keys():
            model_col = f'probability_{name.lower().replace(" ", "_")}'
            if model_col in prediction_data.columns:
                candidates[model_col] = prediction_data[model_col]
    
    if 'kpi_rating' in prediction_data.columns:
        candidates['kpi_rating'] = prediction_data['kpi_rating']
    if 'kpi_category' in prediction_data.columns:
        candidates['kpi_category'] = prediction_data['kpi_category']
    
    candidates = candidates.sort_values('predicted_probability', ascending=False).reset_index(drop=True)
    
    candidates['rank'] = range(1, len(candidates) + 1)
    
    # Compare predictions from different models 
    if all_models and len(all_models) > 1:
        top_n = min(20, len(candidates))
        top_candidates = candidates.head(top_n)
        
        plt.figure(figsize=(16, 8))
        
        model_cols = ['predicted_probability'] + [f'probability_{name.lower().replace(" ", "_")}' for name in all_models.keys() if f'probability_{name.lower().replace(" ", "_")}' in candidates.columns]
        model_names = ['Combined'] + list(all_models.keys())
        
        
        valid_model_cols = [col for col in model_cols if col in top_candidates.columns]
        valid_model_names = [model_names[model_cols.index(col)] for col in valid_model_cols]
        
      
        plot_data = top_candidates.head(10).copy()
        
      
        x = np.arange(len(plot_data))
        width = 0.8 / len(valid_model_cols)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
    
        for i, (col, name) in enumerate(zip(valid_model_cols, valid_model_names)):
            offset = width * i - width * len(valid_model_cols) / 2 + width / 2
            bars = ax.bar(x + offset, plot_data[col], width, label=name)
            
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.set_xlabel('Player')
        ax.set_ylabel('Probability')
        ax.set_title(f'Model Comparison: Top 10 Ballon d\'Or {year} Candidates', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data[player_col], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    
    plt.figure(figsize=(16, 8))
    
    top20 = candidates.head(20)
    
    if 'kpi_rating' in candidates.columns:
        colors = plt.cm.viridis(top20['kpi_rating'] / 100)
        bars = plt.bar(top20[player_col], top20['predicted_probability'], color=colors)
    else:
        bars = plt.bar(top20[player_col], top20['predicted_probability'], color='lightblue')
    
    for i in range(min(3, len(bars))):
        bars[i].set_color(['gold', 'silver', '#cd7f32'][i])
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
        
        if 'kpi_rating' in top20.columns:
            kpi = top20.iloc[i]['kpi_rating']
            plt.annotate(f'KPI: {kpi}',
                        xy=(bar.get_x() + bar.get_width() / 2, height/2),
                        ha='center',
                        fontsize=9,
                        color='white')
    
    plt.xlabel('Player')
    plt.ylabel('Probability of Winning Ballon d\'Or')
    plt.title(f'Top 20 Predicted Ballon d\'Or {year} Candidates with KPI Ratings', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, min(1, top20['predicted_probability'].max() * 1.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 predicted Ballon d'Or {year} candidates:")
    for i, (_, row) in enumerate(candidates.head(10).iterrows()):
        kpi_info = f" - KPI: {row['kpi_rating']} ({row['kpi_category']})" if 'kpi_rating' in row else ""
        print(f"{i+1}. {row[player_col]} - {row['predicted_probability']:.1%}{kpi_info}")
    
    return candidates

def main(file_path="FOOT.csv"):
    try:
        feature_df = pd.DataFrame()
        model = None
        candidates = pd.DataFrame()
        feature_cols = []
        all_models = {}

        print(f"Loading data from {file_path}...")
        df = load_csv(file_path)
        if df.empty:
            print(f"Error: Could not load data from {file_path} or file is empty")
            return feature_df, model, candidates, feature_cols, all_models

        print("Processing player data...")
        processed_df = process_player_data(df)
        
        print("Creating Ballon d'Or labels...")
        labeled_df = create_ballon_dor_labels(processed_df)
        winner_count = labeled_df['ballon_dor_winner'].sum()
        print(f"Identified {winner_count} Ballon d'Or winners in dataset")
        
        print("Calculating KPI ratings...")
        kpi_df = calculate_kpi_rating(labeled_df)
        
        print("Visualizing KPI data...")
        visualize_kpi_data(kpi_df)
        
        print("Building prediction models...")
        model, feature_cols, importance_df, all_models = build_prediction_model(kpi_df)

        if model is None:
            print("Error: Failed to build prediction models")
            return kpi_df, None, pd.DataFrame(), feature_cols, {}

        print("Predicting Ballon d'Or candidates...")
        candidates = predict_ballon_dor_candidates(kpi_df, model, feature_cols, all_models=all_models)
        
        print("Analysis complete!")
        return kpi_df, model, candidates, feature_cols, all_models

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None, pd.DataFrame(), [], {}

        print("dsedfs")
        
if __name__ == "__main__":
    main()