import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import os

class PlayerPropModel:
    """Ridge regression model for player prop predictions with enhanced features"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
        # One model for each stat type
        self.stat_types = ['PTS', 'AST', 'REB', 'FG3M', 'PRA']
    
    def prepare_training_data(self, features_df, target_stat):
        """Prepare features and target for training"""
        X = features_df.copy()
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Target is the actual stat value
        y = X[target_stat] if target_stat in X.columns else None
        
        # Remove target from features
        feature_cols = [col for col in X.columns if col not in self.stat_types]
        X = X[feature_cols]
        
        return X, y
    
    def train(self, features_df, stat_type):
        """Train ridge regression model for a specific stat"""
        X, y = self.prepare_training_data(features_df, stat_type)
        
        if y is None or len(y) == 0:
            print(f"No training data for {stat_type}")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train ridge model
        model = Ridge(alpha=self.alpha)
        model.fit(X_scaled, y)
        
        # Store model and scaler
        self.models[stat_type] = model
        self.scalers[stat_type] = scaler
        
        print(f"Trained {stat_type} model - R² score: {model.score(X_scaled, y):.3f}")
    
    def predict(self, features_dict, stat_type):
        """Make prediction for a single player"""
        if stat_type not in self.models:
            # Fallback to enhanced prediction if model not trained
            return self._fallback_prediction(features_dict, stat_type)
        
        # Convert features to DataFrame
        feature_cols = [col for col in self.feature_names if col not in self.stat_types]
        X = pd.DataFrame([features_dict])[feature_cols]
        
        # Fill missing features with 0
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Scale and predict
        X_scaled = self.scalers[stat_type].transform(X)
        prediction = self.models[stat_type].predict(X_scaled)[0]
        
        # Apply shrinkage toward season average (reduce variance)
        season_avg_key = f'{stat_type}_avg'
        if season_avg_key in features_dict:
            season_avg = features_dict[season_avg_key]
            # 70% model, 30% season average
            prediction = 0.7 * prediction + 0.3 * season_avg
        
        return max(0, prediction)  # Can't predict negative stats
    
    def _fallback_prediction(self, features_dict, stat_type):
        """
        Enhanced fallback prediction with:
        - Head-to-head history weighting
        - Recent opponent form
        - Positional adjustments
        """
        # Base averages
        season_key = f'{stat_type}_avg'
        last5_key = f'{stat_type}_last5'
        last10_key = f'{stat_type}_last10'
        h2h_key = f'h2h_{stat_type}_avg'
        
        season_avg = features_dict.get(season_key, 0)
        last5_avg = features_dict.get(last5_key, season_avg)
        last10_avg = features_dict.get(last10_key, season_avg)
        h2h_avg = features_dict.get(h2h_key, 0)
        h2h_games = features_dict.get(f'h2h_{stat_type}_games', 0)
        
        # Weighted prediction based on data availability
        if h2h_games >= 5:
            # Strong head-to-head history: weight it heavily
            base_prediction = 0.4 * h2h_avg + 0.35 * last5_avg + 0.15 * last10_avg + 0.10 * season_avg
        elif h2h_games >= 2:
            # Some head-to-head history: moderate weight
            base_prediction = 0.25 * h2h_avg + 0.40 * last5_avg + 0.25 * last10_avg + 0.10 * season_avg
        else:
            # No head-to-head: use recent form heavily
            base_prediction = 0.50 * last5_avg + 0.30 * last10_avg + 0.20 * season_avg
        
        # Opponent adjustments - now using recent form
        season_def_rating = features_dict.get('opp_def_rating', 110)
        recent_def_rating = features_dict.get('opp_recent_def_rating', season_def_rating)
        def_trend = features_dict.get('opp_def_trend', 0)
        
        # Use recent defense if significantly different from season average
        if abs(recent_def_rating - season_def_rating) > 5:
            # Recent form is significantly different, weight it more
            effective_def_rating = 0.7 * recent_def_rating + 0.3 * season_def_rating
        else:
            # Similar, use season average
            effective_def_rating = season_def_rating
        
        league_avg_def = 110
        
        # Stat-specific adjustments
        if stat_type == 'PTS':
            def_adjustment = 1.0 + ((effective_def_rating - league_avg_def) / league_avg_def) * 0.20
        elif stat_type == 'AST':
            def_adjustment = 1.0 + ((effective_def_rating - league_avg_def) / league_avg_def) * 0.12
        elif stat_type == 'REB':
            def_adjustment = 1.0 + ((effective_def_rating - league_avg_def) / league_avg_def) * 0.08
        elif stat_type == 'FG3M':
            def_adjustment = 1.0 + ((effective_def_rating - league_avg_def) / league_avg_def) * 0.15
        elif stat_type == 'PRA':
            def_adjustment = 1.0 + ((effective_def_rating - league_avg_def) / league_avg_def) * 0.15
        else:
            def_adjustment = 1.0
        
        # Cap adjustment to reasonable range
        def_adjustment = max(0.80, min(1.20, def_adjustment))
        
        # Apply head-to-head trend adjustment
        h2h_trend = features_dict.get(f'h2h_{stat_type}_trend', 0)
        if abs(h2h_trend) > 2:  # Significant trend
            trend_adjustment = 1.0 + (h2h_trend / base_prediction) * 0.1 if base_prediction > 0 else 1.0
            trend_adjustment = max(0.95, min(1.05, trend_adjustment))
        else:
            trend_adjustment = 1.0
        
        prediction = base_prediction * def_adjustment * trend_adjustment
        
        # Rest days adjustment
        rest_days = features_dict.get('rest_days', 3)
        is_b2b = features_dict.get('is_back_to_back', 0)
        
        if is_b2b:
            prediction = prediction * 0.93  # Bigger penalty for back-to-back
        elif rest_days >= 3:
            prediction = prediction * 1.03  # Boost for rest
        
        return max(0, prediction)
    
    def predict_double_double(self, features_dict):
        """Predict probability of double-double with enhanced features"""
        # Use historical double-double rate with adjustments
        base_prob = features_dict.get('dd_probability', 0.0)
        
        # Adjust based on recent form
        pts_last5 = features_dict.get('PTS_last5', 0)
        reb_last5 = features_dict.get('REB_last5', 0)
        ast_last5 = features_dict.get('AST_last5', 0)
        
        # Boost probability if multiple stats trending high
        high_stats = sum([pts_last5 >= 10, reb_last5 >= 8, ast_last5 >= 8])
        if high_stats >= 2:
            base_prob = min(base_prob * 1.3, 0.95)
        
        # Opponent adjustment
        opp_def = features_dict.get('opp_recent_def_rating', features_dict.get('opp_pts_allowed', 110))
        if opp_def > 115:  # Weak defense
            base_prob = min(base_prob * 1.15, 0.95)
        elif opp_def < 105:  # Strong defense
            base_prob = base_prob * 0.88
        
        # Head-to-head adjustment
        h2h_pts = features_dict.get('h2h_PTS_avg', 0)
        h2h_reb = features_dict.get('h2h_REB_avg', 0)
        h2h_ast = features_dict.get('h2h_AST_avg', 0)
        h2h_games = features_dict.get('h2h_PTS_games', 0)
        
        if h2h_games >= 3:
            # Check historical double-double rate vs this team
            h2h_dd_stats = sum([h2h_pts >= 10, h2h_reb >= 8, h2h_ast >= 8])
            if h2h_dd_stats >= 2:
                base_prob = min(base_prob * 1.2, 0.95)
        
        return base_prob
    
    def save_model(self, filepath):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'alpha': self.alpha
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained models from disk"""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.alpha = model_data['alpha']
        
        print(f"Model loaded from {filepath}")
        return True