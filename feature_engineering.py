import pandas as pd
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FeatureEngineering():
    # Feature engineering utilities for dataset preprocessing and analysis.
    def __init__(self, df):
        # Initialize with a DataFrame.
        self.df = df
        self.selected_features = None
        self.features_score_df = None
        
    def drop_constant_features(self):
        # Remove features with no variance.
        # Returns a list of dropped columns.
        constant_cols = self.df.columns[self.df.nunique() == 1]
        numeric_df = self.df.select_dtypes(['number'])
        zero_std_cols = numeric_df.columns[numeric_df.std() == 0]
        to_drop = list(set(constant_cols) | set(zero_std_cols))
        self.df.drop(columns=to_drop, inplace=True)
        return to_drop
    
    def feature_importance(self):
        # Select top features using SelectKBest (f_classif).
        # Stores results as class attributes.
        self.df = self.df.dropna(subset=['flow_bytes_s'])
        
        self.X = self.df.drop(columns='target_encoded')
        self.y = self.df['target_encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        SAMPLE_SIZE = 10000  
        sample_size = min(SAMPLE_SIZE, len(self.X_train))
        sample_idx = self.X_train.sample(sample_size, random_state=42).index
        X_sample = self.X_train.loc[sample_idx]
        y_sample = self.y_train.loc[sample_idx]
        
        selector = SelectKBest(score_func=f_classif, k=50) 
        selector.fit(X_sample, y_sample)
        
        selected_features_mask = selector.get_support()
        selected_feature_names = self.X.columns[selected_features_mask].tolist()
        self.selected_features = selected_feature_names
        feature_scores = selector.scores_

        features_score_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'score': feature_scores
        })

        features_score_df = features_score_df[features_score_df['feature'].isin(self.selected_features)]
        features_score_df = features_score_df.sort_values(by='score', ascending=False)
        self.features_score_df = features_score_df
    
    def print_feature_summary(self):
        # Print a summary table of top features and their scores.
        print(self.features_score_df)
        print('*' * 50)
        print(
            "The chart displays the top 30 features according to f_classif score. "
            "Although all recommended features were included in the neural network, "
            "the visualization is limited to the 30 most important features for better clarity and interpretability."
        )
        print('*' * 50)

    def plot_feature_importance(self, threshold=30, save_path='feature_importance.png'):
        # Plot and save feature importance chart above threshold.
        
        selected_features_df = self.features_score_df[self.features_score_df['score'] > threshold]
        
        plt.figure(figsize=(12, 6))
        plt.barh(selected_features_df['feature'], selected_features_df['score'])
        plt.xlabel('Score')
        plt.title(f'Feature Importance (f_classif) - All Features with Score > {threshold}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_graph.png')
        try:
            plt.show()  
        except:
            pass  
        plt.close()

        
        

        
        