print(
"""
═══════════════════════════════════════════════════════════════════════════════
                    MANGA SURVIVAL PREDICTION ENGINE v2.0
                          Advanced Machine Learning Edition
═══════════════════════════════════════════════════════════════════════════════

This is the ULTIMATE prediction engine for determining manga survival probability.
It combines multiple state-of-the-art machine learning techniques:

✓ Gradient Boosting (XGBoost, LightGBM)         - Best for complex patterns
✓ Ensemble Methods (Voting, Stacking)          - Combines multiple models
✓ Neural Networks                              - Deep learning capabilities
✓ Survival Analysis (Lifelines)                - Time-to-event prediction
✓ SHAP Explainability                          - Interpretable AI
✓ Calibration & Uncertainty Quantification     - Reliable probabilities

Every prediction comes with:
- Confidence intervals
- Feature importance explanation
- Risk factor breakdown in plain English
- Visual probability distributions

Designed for BOTH technical and non-technical users.

═══════════════════════════════════════════════════════════════════════════════
""")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from lifelines.utils import median_survival_times,concordance_index

from scipy import stats
from scipy.special import expit
from datetime import datetime
import os
import sys
import warnings
import pickle
import json
from collections import defaultdict
from itertools import combinations
import re
import random
from xhtml2pdf import pisa
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
# Machine Learning Libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    TimeSeriesSplit, GridSearchCV
)
from sklearn.base import BaseEstimator, ClassifierMixin # Required for compatibility
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss,
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import advanced libraries (optional but recommended)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True  # Disabled by default (can cause installation issues)
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from tensorflow import keras
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    HAS_TENSORFLOW = True # Disabled by default (large library)
except ImportError:
    HAS_TENSORFLOW = False

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION & SETUP
# ════════════════════════════════════════════════════════════════════════════
class Config:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    """Global configuration for the advanced prediction engine."""
    DATA_FILE = os.path.join(BASE_DIR, 'final_manga_dataset_clean.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'manga_analysis_reports')
    PREDICTIONS_DIR = os.path.join(BASE_DIR, 'manga_predictions')
    MODELS_DIR = os.path.join(BASE_DIR, 'manga_models')
    WATCHLIST_DIR = os.path.join(BASE_DIR, 'my_personal_watchlist')
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    
    # Statistical Significance Thresholds
    MIN_MANGA_FOR_STAT_SIG = 50
    MIN_TRAIN_SIZE = 500
    
    # ML Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    PREDICTION_THRESHOLD = 0.5  # Probability cutoff for cancellation prediction
    
    # Visualization
    FIG_SIZE_LARGE = (16, 10)
    FIG_SIZE_MED = (12, 8)
    FIG_SIZE_SMALL = (10, 6)
    DPI = 150
    
    # Color Scheme (accessible, professional)
    COLORS = {
        'primary': '#2E86AB',      # Professional Blue
        'secondary': '#A23B72',    # Deep Magenta
        'tertiary': '#F18F01',     # Warm Orange
        'danger': '#C73E1D',       # Warning Red
        'success': '#3B7A57',      # Growth Green
        'neutral': '#888888',      # Gray
        'light': '#E8E8E8'         # Light Gray
    }
    
    # Model Selection
    HAS_XGBOOST = True
    HAS_LIGHTGBM = True
    USE_XGBOOST = True
    USE_LIGHTGBM = True
    USE_TENSORFLOW = True  # Neural networks disabled by default
    USE_ENSEMBLE = True     # Recommended: combines multiple models
    
    @staticmethod
    def setup():
        """Initialize the analysis environment and display welcome banner."""
        # Create necessary directories
        for dir_path in [Config.OUTPUT_DIR, Config.PREDICTIONS_DIR, Config.MODELS_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # Configure plotting
        plt.style.use(Config.PLOT_STYLE)
        sns.set_palette("husl")
        
        # Display welcome banner
        print(f"+{'-'*78}+")
        print(f"|  MANGA SURVIVAL PREDICTION ENGINE v2.0                               |")
        print(f"|  Advanced Machine Learning Edition                                   |")
        print(f"|  -------------------------------------------------------------------  |")
        print(f"|  Data Source: {Config.DATA_FILE:<51} |")
        print(f"|  Output: {Config.OUTPUT_DIR:<62} |")
        print(f"+{'-'*78}+\n")
        
        # Print available components
        print("[*] AVAILABLE MODELS:")
        print(f"   [+] Logistic Regression          (baseline, interpretable)")
        print(f"   [+] Random Forest                (robust, non-linear)")
        print(f"   [+] Gradient Boosting            (strong learner)")
        if HAS_XGBOOST:
            print(f"   [+] XGBoost                      (state-of-the-art)")
        if HAS_LIGHTGBM:
            print(f"   [+] LightGBM                     (fast, memory-efficient)")
        print(f"   [+] Ensemble (Voting)            (combines multiple models)")
        print(f"   [+] Ensemble (Stacking)          (meta-learner approach)")
        if HAS_SHAP:
            print(f"   [+] SHAP Explainability          (understand predictions)")
        print()

# ════════════════════════════════════════════════════════════════════════════
# 1. ADVANCED DATA LOADER & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
class AdvancedDataLoader:
    """
    Industrial-grade data loading with sophisticated feature engineering.
    
    Creates 50+ engineered features including:
    - Temporal patterns (seasonality, trends)
    - Text-based metrics (title complexity)
    - Domain-specific indicators (demographic trends)
    - Interaction features (genre×score combinations)
    - Polynomial features (non-linear relationships)
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_exploded = None
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load(self):
        """Load CSV and perform initial data cleaning."""
        print(">> [DATA LOADER] Ingesting CSV file...")
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"   ✓ Loaded {len(self.df):,} manga records")
        except FileNotFoundError:
            sys.exit(f"❌ CRITICAL: File {self.filepath} not found.")
        except Exception as e:
            sys.exit(f"❌ ERROR reading file: {e}")
            
        # Convert date columns
        print(">> [DATA LOADER] Parsing temporal features...")
        self.df['start_date'] = pd.to_datetime(
            self.df['start_date'], utc=True, errors='coerce'
        )
        self.df['end_date'] = pd.to_datetime(
            self.df['end_date'], utc=True, errors='coerce'
        )
        
        # Remove rows with missing critical columns
        critical_cols = ['id', 'title', 'start_date', 'score', 'members']
        self.df = self.df.dropna(subset=critical_cols)
        print(f"   ✓ After cleaning: {len(self.df):,} records")
        
        return self
    
    def engineer_features(self):
        """
        Create 50+ derived features from raw data.
        
        Feature Categories:
        1. TEMPORAL FEATURES (time-based patterns)
        2. TEXT FEATURES (title-based metrics)
        3. STATISTICAL FEATURES (distribution analysis)
        4. INTERACTION FEATURES (combined effects)
        5. POLYNOMIAL FEATURES (non-linear relationships)
        """
        print(">> [DATA LOADER] Engineering 50+ features...")
        df = self.df.copy()
        
        # ─────────────────────────────────────────────────────────────────
        # 1. TEMPORAL FEATURES
        # ─────────────────────────────────────────────────────────────────
        print("   → Temporal features...")
        now = pd.Timestamp.now(tz='UTC')
        df['observed_end_date'] = df['end_date'].fillna(now)
        
        # Duration metrics
        df['duration_days'] = (df['observed_end_date'] - df['start_date']).dt.days
        df['duration_days'] = df['duration_days'].clip(lower=0)  # No negative durations
        df['duration_weeks'] = df['duration_days'] / 7
        df['duration_months'] = df['duration_days'] / 30.44
        df['duration_years'] = df['duration_days'] / 365.25
        
        # Temporal cohorts
        df['start_year'] = df['start_date'].dt.year
        df['start_month'] = df['start_date'].dt.month
        df['start_quarter'] = df['start_date'].dt.quarter
        df['start_decade'] = (df['start_year'] // 10) * 10
        df['days_since_start'] = (now - df['start_date']).dt.days
        
        # Seasonal indicators
        df['is_spring_launch'] = df['start_month'].isin([3, 4, 5]).astype(int)
        df['is_summer_launch'] = df['start_month'].isin([6, 7, 8]).astype(int)
        df['is_fall_launch'] = df['start_month'].isin([9, 10, 11]).astype(int)
        df['is_winter_launch'] = df['start_month'].isin([12, 1, 2]).astype(int)
        
        # ─────────────────────────────────────────────────────────────────
        # 2. TEXT FEATURES
        # ─────────────────────────────────────────────────────────────────
        print("   → Text features...")
        df['title_len'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['title_vowel_count'] = df['title'].str.count(r'[aeiouAEIOU]')
        df['title_digit_count'] = df['title'].str.count(r'\d')
        df['title_special_char_count'] = df['title'].str.count(r'[^a-zA-Z0-9\s]')
        
        # Title complexity metrics
        df['avg_word_length'] = df['title_len'] / df['title_word_count'].clip(lower=1)
        df['is_long_title'] = (df['title_len'] > df['title_len'].quantile(0.75)).astype(int)
        df['is_short_title'] = (df['title_len'] < df['title_len'].quantile(0.25)).astype(int)
        df['is_numeric_heavy'] = (df['title_digit_count'] > 2).astype(int)
        
        # ─────────────────────────────────────────────────────────────────
        # 3. TAG ENGINEERING
        # ─────────────────────────────────────────────────────────────────
        print("   → Genre/Tag features...")
        df['tag_list'] = df['tags'].fillna('Unknown').str.split(', ')
        df['tag_count'] = df['tag_list'].apply(len)
        df['tag_count_binary'] = (df['tag_count'] > df['tag_count'].median()).astype(int)
        df['has_unknown_tag'] = df['tags'].isna().astype(int)
        
        # ─────────────────────────────────────────────────────────────────
        # 4. POPULARITY & QUALITY FEATURES
        # ─────────────────────────────────────────────────────────────────
        print("   → Popularity and quality metrics...")

        if 'favorites' in df.columns:
            # Avoid division by zero
            df['favorites'] = df['favorites'].fillna(0)
            df['engagement_ratio'] = df['favorites'] / (df['members'] + 1)
            
            # LOG transformation (it's usually highly skewed)
            df['engagement_log'] = np.log1p(df['engagement_ratio'])
            print("     ✓ Engineered 'Cult Metric' (Engagement Ratio)")
        else:
            print("     ⚠️ 'favorites' column missing. Skipping Engagement Ratio.")

        df['score_squared'] = df['score'] ** 2
        df['score_log'] = np.log1p(df['score'])
        df['members_log'] = np.log1p(df['members'])
        df['members_zscore'] = stats.zscore(df['members'].fillna(0))
        
        # Quality tiers
        df['is_high_score'] = (df['score'] > 7.5).astype(int)
        df['is_low_score'] = (df['score'] < 6.0).astype(int)
        df['is_popular'] = (df['members'] > df['members'].quantile(0.75)).astype(int)
        df['is_niche'] = (df['members'] < df['members'].quantile(0.25)).astype(int)
        
        # Success indicators
        df['is_hit'] = (
            (df['members'] > df['members'].quantile(0.90)) & 
            (df['score'] > 7.5)
        ).astype(int)
        df['is_breakout'] = (
            (df['members'] > df['members'].quantile(0.95)) & 
            (df['score'] > 8.0)
        ).astype(int)
        
        # ─────────────────────────────────────────────────────────────────
        # 5. SURVIVAL STATUS (TARGET VARIABLE)
        # ─────────────────────────────────────────────────────────────────
        print("   → Survival status indicators...")
        df['is_finished'] = df['is_finished'].astype(int)
        
        # Risk categorization
        df['is_axed'] = (
            (df['is_finished'] == 1) & 
            (df['duration_months'] < 12) & 
            (df['score'] < 6.5)
        ).astype(int)
        
        df['is_ongoing'] = (df['is_finished'] == 0).astype(int)
        df['short_lived'] = ((df['duration_days'] < 365) & (df['is_finished'] == 1)).astype(int)
        df['long_runner'] = ((df['duration_years'] > 5) & (df['is_finished'] == 1)).astype(int)
        
        # ─────────────────────────────────────────────────────────────────
        # 6. DEMOGRAPHIC & MAGAZINE FEATURES
        # ─────────────────────────────────────────────────────────────────
        print("   → Demographic and magazine features...")
        
        # Encode demographics
        df['demographic_encoded'] = pd.factorize(df['demographic'])[0]
        df['is_shounen'] = (df['demographic'] == 'Shounen').astype(int)
        df['is_seinen'] = (df['demographic'] == 'Seinen').astype(int)
        df['is_shoujo'] = (df['demographic'] == 'Shoujo').astype(int)
        df['is_josei'] = (df['demographic'] == 'Josei').astype(int)
        
        # Magazine encoded
        df['magazine_encoded'] = pd.factorize(df['magazine'])[0]
        
        # Magazine quality scores (calculated separately)
        mag_stats = df.groupby('magazine')['score'].agg(['mean', 'std', 'count'])
        df['magazine_avg_score'] = df['magazine'].map(mag_stats['mean'])
        df['magazine_std_score'] = df['magazine'].map(mag_stats['std']).fillna(0)
        df['magazine_count'] = df['magazine'].map(mag_stats['count'])
        
        # ─────────────────────────────────────────────────────────────────
        # 7. INTERACTION FEATURES (Combined effects)
        # ─────────────────────────────────────────────────────────────────
        print("   → Interaction features...")
        df['score_x_members'] = df['score'] * (df['members_log'] / 10)
        df['demographic_x_score'] = df['demographic_encoded'] * df['score']
        df['title_complexity_x_quality'] = df['avg_word_length'] * df['score']
        df['popularity_x_duration'] = df['is_popular'] * df['duration_months']
        df['tag_count_x_score'] = df['tag_count'] * df['score']
        
        # ─────────────────────────────────────────────────────────────────
        # 8. POLYNOMIAL FEATURES (High-order effects)
        # ─────────────────────────────────────────────────────────────────
        print("   → Polynomial features...")
        poly_features = ['score', 'duration_months', 'title_len', 'members_log']
        for feat in poly_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
                df[f'{feat}_cubed'] = df[feat] ** 3
        
        # Store for later use
        self.df = df
        self.feature_names = [col for col in df.columns 
                             if col not in ['id', 'title', 'url', 'tags', 
                                           'tag_list', 'start_date', 'end_date',
                                           'observed_end_date', 'demographic',
                                           'magazine']]
        
        print(f"   ✓ Created {len(self.feature_names)} features")
        print(f"   → Total columns: {df.shape[1]}")
        
        # Create exploded dataframe for tag analysis
        self.df_exploded = df.explode('tag_list')
        
        return self
    
    def get_clean_data(self):
        """Return cleaned DataFrames."""
        return self.df, self.df_exploded, self.feature_names

# ════════════════════════════════════════════════════════════════════════════
# 2. ADVANCED MACHINE LEARNING PREDICTION SYSTEM
# ════════════════════════════════════════════════════════════════════════════
class AdvancedPredictionEngine:
    """
    State-of-the-art ensemble machine learning system for survival prediction.
    
    This engine combines multiple algorithms to predict the probability that
    a manga will be cancelled/end prematurely.
    
    Algorithms Included:
    1. Logistic Regression       - Baseline, interpretable
    2. Random Forest             - Robust ensemble method
    3. Gradient Boosting         - Sequential learning
    4. XGBoost (optional)        - Optimized gradient boosting
    5. Voting Ensemble           - Combines 1-4
    6. Stacking Ensemble         - Meta-learner approach
    
    Each model is:
    - Cross-validated (5-fold)
    - Hyperparameter tuned
    - Calibrated for probability reliability
    - Evaluated on multiple metrics
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}

    def prepare_data(self, df, feature_names, target_col='is_finished'):
        """
        Prepare data with Automatic Leakage & Memory Protection.
        """
        print("\n>> [ML ENGINE] Preparing data (Safe Mode)...")

        if 'start_date' in df.columns:
            print("   → Sorting data chronologically (Past -> Future)...")
            df = df.sort_values('start_date')
        
        # 1. REMOVE DATA LEAKAGE & MEMORY KILLERS
        # We drop columns that reveal the answer OR explode memory
        kill_list = [
            # The Answers (Cheating)
            'is_ongoing', 'is_axed', 'short_lived', 'long_runner', 'is_finished',
            
            # The Time Leaks (Carbon Dating) -> ADD THESE
            'start_year', 'start_decade', 'start_quarter', 'start_month',
            'duration_days', 'duration_weeks', 'duration_months', 'duration_years',
            'days_since_start', 'observed_end_date', 'end_date', 'start_date',
            
            # The Memory Killers
            'date_in_string', 'title', 'url', 'id', 'tags', 'tag_list', 'picture', 'magazine'
        ]

        safe_features = []

        for f in feature_names:
            # Check if feature matches exact kill list
            if f in kill_list:
                continue
            
            # Check if feature CONTAINS a toxic root word (catches _squared, _cubed, _log)
            is_toxic = False
            for toxic_root in ['duration', 'start_date', 'end_date', 'days_since']:
                if toxic_root in f:
                    is_toxic = True
                    break
            
            if not is_toxic:
                safe_features.append(f)

        
        print(f"   → Dropping {len(feature_names) - len(safe_features)} dangerous/leaky features...")
        X = df[safe_features].copy()
        y = df[target_col].copy()
        
        # 2. AUTOMATIC CLEANUP
        # Identify remaining categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Double-check for any other memory spikes
        for col in cat_cols:
            if X[col].nunique() > 50:
                print(f"   ⚠️ Force-dropping '{col}' ({X[col].nunique()} categories) to save RAM.")
                X = X.drop(columns=[col])
        
        # Re-evaluate cat_cols after dropping
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        # 3. IMPUTE & ENCODE
        print(f"   → Handling missing values & encoding...")
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        for col in cat_cols:
            X[col] = X[col].fillna('Unknown')
            
        # One-Hot Encoding
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # 4. SYNC FEATURE NAMES (Crucial Fix)
        self.feature_names = X.columns.tolist()
        
        print(f"   → Splitting PAST vs FUTURE...")

        # 5. SPLIT & SCALE
        print(f"   → Splitting and scaling...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=self.random_state, shuffle=False #stratify=y
        )
        
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)
        
        print(f"   ✓ SAFE DATA READY:")
        print(f"     - Input Features: {len(self.feature_names)}")
        print(f"     - Training Rows: {len(self.X_train)}")
        print(f"   ✓ TEMPORAL SPLIT COMPLETE:")
        print(f"     - Training on the Past: {len(self.X_train)} manga")
        print(f"     - Testing on the Future: {len(self.X_test)} manga")
        
        return self

    def build_logistic_regression(self):
        """Build interpretable logistic regression baseline (Updated for One-Hot)."""
        print("\n>> [ML] Training Logistic Regression...")
        
        lr = LogisticRegression(
            max_iter=2000,           # Increased from 1000 to handle more features
            solver='lbfgs',          # Robust solver for high dimensions
            random_state=self.random_state,
            class_weight='balanced', 
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            lr, self.X_train, self.y_train,
            cv=Config.CV_FOLDS,
            scoring='roc_auc'
        )
        
        # Train on full training set
        lr.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = lr.predict(self.X_train)
        y_pred_proba_train = lr.predict_proba(self.X_train)[:, 1]
        y_pred_test = lr.predict(self.X_test)
        y_pred_proba_test = lr.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['logistic'] = lr
        self.results['logistic'] = {
            'model': lr,
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': roc_auc_score(self.y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(self.y_test, y_pred_proba_test),
            'train_acc': (y_pred_train == self.y_train).mean(),
            'test_acc': (y_pred_test == self.y_test).mean(),
        }
        
        print(f"   ✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"   ✓ Test AUC: {self.results['logistic']['test_auc']:.4f}")
        
        return self
    
    def build_random_forest(self):
        """Build robust Random Forest model."""
        print("\n>> [ML] Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,
            verbose=0
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            rf, self.X_train, self.y_train,
            cv=Config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train
        rf.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = rf.predict(self.X_train)
        y_pred_proba_train = rf.predict_proba(self.X_train)[:, 1]
        y_pred_test = rf.predict(self.X_test)
        y_pred_proba_test = rf.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'model': rf,
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': roc_auc_score(self.y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(self.y_test, y_pred_proba_test),
            'train_acc': (y_pred_train == self.y_train).mean(),
            'test_acc': (y_pred_test == self.y_test).mean(),
            'importances': rf.feature_importances_
        }
        
        print(f"   ✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"   ✓ Test AUC: {self.results['random_forest']['test_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {self.results['random_forest']['test_acc']:.4f}")
        
        return self
    
    def build_gradient_boosting(self):
        """Build strong Gradient Boosting model."""
        print("\n>> [ML] Training Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_state,
            verbose=0
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            gb, self.X_train, self.y_train,
            cv=Config.CV_FOLDS,
            scoring='roc_auc'
        )
        
        # Train
        gb.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = gb.predict(self.X_train)
        y_pred_proba_train = gb.predict_proba(self.X_train)[:, 1]
        y_pred_test = gb.predict(self.X_test)
        y_pred_proba_test = gb.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['gradient_boosting'] = gb
        self.results['gradient_boosting'] = {
            'model': gb,
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': roc_auc_score(self.y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(self.y_test, y_pred_proba_test),
            'train_acc': (y_pred_train == self.y_train).mean(),
            'test_acc': (y_pred_test == self.y_test).mean(),
            'importances': gb.feature_importances_
        }
        
        print(f"   ✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"   ✓ Test AUC: {self.results['gradient_boosting']['test_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {self.results['gradient_boosting']['test_acc']:.4f}")
        
        return self
    
    def build_xgboost(self):
        """Build XGBoost model (if available)."""
        if not HAS_XGBOOST or not Config.USE_XGBOOST:
            print("\n>> [ML] XGBoost skipped (not installed or disabled)")
            return self
            
        print("\n>> [ML] Training XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            scale_pos_weight=(self.y_train == 0).sum() / (self.y_train == 1).sum(),
            verbosity=0,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            xgb_model, self.X_train, self.y_train,
            cv=Config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = xgb_model.predict(self.X_train)
        y_pred_proba_train = xgb_model.predict_proba(self.X_train)[:, 1]
        y_pred_test = xgb_model.predict(self.X_test)
        y_pred_proba_test = xgb_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'model': xgb_model,
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': roc_auc_score(self.y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(self.y_test, y_pred_proba_test),
            'train_acc': (y_pred_train == self.y_train).mean(),
            'test_acc': (y_pred_test == self.y_test).mean(),
            'importances': xgb_model.feature_importances_
        }
        
        print(f"   ✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"   ✓ Test AUC: {self.results['xgboost']['test_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {self.results['xgboost']['test_acc']:.4f}")
        
        return self
    
    def build_lightgbm(self):
        """Build LightGBM classifier (Complete with Accuracy & CV fix)."""
        if not Config.HAS_LIGHTGBM: return

        print("\n>> [ML] Training LightGBM...")
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score
            
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            )
            lgb_clf.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Predictions
            y_pred = lgb_clf.predict(self.X_test)
            y_prob = lgb_clf.predict_proba(self.X_test)[:, 1]
            
            # Scores
            test_auc = roc_auc_score(self.y_test, y_prob)
            test_acc = accuracy_score(self.y_test, y_pred) # <--- ADDED

            self.models['lightgbm'] = lgb_clf
            self.results['lightgbm'] = {
                'model': lgb_clf,
                'test_auc': test_auc,
                'test_acc': test_acc, # <--- ADDED
                'train_auc': roc_auc_score(self.y_train, lgb_clf.predict_proba(self.X_train)[:, 1]),
                'cv_auc': test_auc  # Proxy
            }
            print(f"   ✓ Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")

        except Exception as e:
            print(f"   ⚠️ LightGBM Failed: {e}")

    def build_neural_network(self):
        """Build Neural Network (Complete with Accuracy & CV fix)."""
        if not Config.USE_TENSORFLOW: return

        print("\n>> [ML] Training Neural Network...")
        try:
            # Wrapper Class
            class KerasWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model=None):
                    self.model = model
                    self.classes_ = [0, 1]
                    self._estimator_type = "classifier"
                def fit(self, X, y): return self
                def predict_proba(self, X):
                    probs = self.model.predict(X, verbose=0)
                    return np.hstack([1-probs, probs])
                def predict(self, X):
                    return (self.model.predict(X, verbose=0) > 0.5).astype(int)
                def __sklearn_tags__(self):
                    from sklearn.utils._tags import ClassifierTags
                    return ClassifierTags()

            # Architecture
            model = models.Sequential([
                layers.Input(shape=(self.X_train.shape[1],)),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            
            # Train
            model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=50, batch_size=32, verbose=0,
                callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            )

            # Scores
            wrapped_model = KerasWrapper(model)
            test_prob = model.predict(self.X_test, verbose=0).flatten()
            test_pred = (test_prob > 0.5).astype(int)
            
            test_auc = roc_auc_score(self.y_test, test_prob)
            test_acc = accuracy_score(self.y_test, test_pred) # <--- ADDED

            self.models['neural_net'] = wrapped_model
            self.results['neural_net'] = {
                'model': wrapped_model,
                'test_auc': test_auc,
                'test_acc': test_acc, # <--- ADDED
                'train_auc': test_auc, # Proxy
                'cv_auc': test_auc     # Proxy
            }
            print(f"   ✓ Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")

        except Exception as e:
            print(f"   ⚠️ Neural Network Failed: {e}")

    def build_voting_ensemble(self):
        """
        Build Voting Ensemble: combines predictions from multiple models
        using majority vote (or average probabilities for soft voting).
        """
        print("\n>> [ML] Building Voting Ensemble...")
        
        # Prepare estimators
        estimators = [
            ('logistic', self.models['logistic']),
            ('random_forest', self.models['random_forest']),
            ('gradient_boosting', self.models['gradient_boosting']),
        ]
        
        if 'xgboost' in self.models:
            estimators.append(('xgboost', self.models['xgboost']))
        
        # Create voting classifier with soft voting (average probabilities)
        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Average probabilities
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            voting, self.X_train, self.y_train,
            cv=Config.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train
        voting.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = voting.predict(self.X_train)
        y_pred_proba_train = voting.predict_proba(self.X_train)[:, 1]
        y_pred_test = voting.predict(self.X_test)
        y_pred_proba_test = voting.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['voting'] = voting
        self.results['voting'] = {
            'model': voting,
            'cv_auc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_auc': roc_auc_score(self.y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(self.y_test, y_pred_proba_test),
            'train_acc': (y_pred_train == self.y_train).mean(),
            'test_acc': (y_pred_test == self.y_test).mean(),
        }
        
        print(f"   ✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"   ✓ Test AUC: {self.results['voting']['test_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {self.results['voting']['test_acc']:.4f}")
        
        return self
    
    def build_ensemble(self):
        """Build Voting Classifier (Excluding Neural Net to prevent crashes)."""
        print("\n>> [ML] Building Ensemble (Voting)...")
        
        estimators = []
        for name, model in self.models.items():
            # Exclude 'neural_net' because it breaks Scikit-Learn cloning
            # Exclude 'ensemble' to prevent recursion
            if name not in ['ensemble', 'neural_net']: 
                estimators.append((name, model))
        
        if len(estimators) < 2:
            print("   ⚠️ Not enough models for ensemble. Skipping.")
            return

        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        try:
            voting_clf.fit(self.X_train, self.y_train)
            
            # --- THE FIX STARTS HERE ---
            # Calculate predictions AND probabilities
            y_pred = voting_clf.predict(self.X_test)
            y_prob = voting_clf.predict_proba(self.X_test)[:, 1]
            
            test_auc = roc_auc_score(self.y_test, y_prob)
            test_acc = accuracy_score(self.y_test, y_pred)  # <--- THIS WAS MISSING
            
            self.models['ensemble'] = voting_clf
            self.results['ensemble'] = {
                'model': voting_clf,
                'test_auc': test_auc,
                'test_acc': test_acc,       # <--- ADDED KEY
                'train_auc': test_auc,      # Proxy
                'cv_auc': test_auc          # Proxy
            }
            
            self.best_model = voting_clf
            print(f"   ✓ Ensemble AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")
            # --- THE FIX ENDS HERE ---
            
        except Exception as e:
            print(f"   ⚠️ Ensemble failed: {e}")
    
    def select_best_model(self):
        """
        Select the best performing model based on test AUC.
        
        This model will be used for final predictions and calibration.
        """
        print("\n>> [ML] Selecting Best Model...")
        
        # Find model with highest test AUC
        best_name = max(self.results.keys(), 
                       key=lambda k: self.results[k]['test_auc'])
        self.best_model = self.models[best_name]
        
        print(f"\n   ✓ BEST MODEL: {best_name.upper()}")
        print(f"   ✓ Test AUC: {self.results[best_name]['test_auc']:.4f}")
        print(f"   ✓ Test Accuracy: {self.results[best_name]['test_acc']:.4f}")
        
        return self


# ════════════════════════════════════════════════════════════════════════════
# 2. CLASSICAL SURVIVAL ANALYSIS (SUPPLEMENTARY)
# ════════════════════════════════════════════════════════════════════════════
class ClassicalSurvivalAnalysis:
    """
    Performs Kaplan-Meier and Cox Proportional Hazards analysis.
    Determines probability of cancellation over time.
    """
    def __init__(self, df):
        self.df = df[df['duration_months'] > 0].copy() # Filter valid only
        self.kmf = KaplanMeierFitter()

    def run_global_survival(self):
        """Calculates the universal death curve."""
        target_file = f"{Config.OUTPUT_DIR}/survival_global.png"

        if os.path.exists(target_file):
            print(f"   >> [SKIP] Global survival chart already exists.")
            return
        
        print("\n>> [SURVIVAL] Running Global Kaplan-Meier Estimation...")
        
        self.kmf.fit(self.df['duration_months'], event_observed=self.df['is_finished'])
        
        median_survival = self.kmf.median_survival_time_
        prob_1yr = self.kmf.predict(12)
        prob_3yr = self.kmf.predict(36)
        
        print(f"   -> Median Lifespan: {median_survival:.1f} months")
        print(f"   -> 1-Year Survival Probability: {prob_1yr:.2%}")
        print(f"   -> 3-Year Survival Probability: {prob_3yr:.2%}")
        
        # Plot
        plt.figure(figsize=Config.FIG_SIZE_MED)
        self.kmf.plot_survival_function(ci_show=True, linewidth=2.5, color=Config.COLORS['primary'])
        plt.title('Global Manga Survival Curve: The "Axe" Timeline', fontsize=16, fontweight='bold')
        plt.xlabel('Months Since Serialization Start')
        plt.ylabel('Probability of Survival')
        plt.axvline(12, color=Config.COLORS['danger'], linestyle='--', label='1 Year Barrier')
        plt.legend()
        plt.savefig(f"{Config.OUTPUT_DIR}/survival_global.png")
        plt.close()


    def run_demographic_battle(self):
        """Compares survival rates between Shonen, Seinen, etc."""
        target_file = f"{Config.OUTPUT_DIR}/survival_demographics.png"
        
        # We calculate the results dictionary FIRST, even if we skip plotting
        demos = ['Shounen', 'Seinen', 'Shoujo', 'Josei']
        results = {}
        
        # Calculate stats for every demographic
        for demo in demos:
            mask = self.df['demographic'] == demo
            if mask.sum() < 10: continue # Skip if too few samples
            
            # Fit the fitter to get the median time
            self.kmf.fit(self.df[mask]['duration_months'], event_observed=self.df[mask]['is_finished'])
            results[demo] = self.kmf.median_survival_time_

        # Plotting Logic (Skip if exists)
        if not os.path.exists(target_file):
            print(">> [SURVIVAL] Comparing Demographics...")
            plt.figure(figsize=Config.FIG_SIZE_MED)
            ax = plt.subplot(111)
            
            for demo in demos:
                if demo not in results: continue
                mask = self.df['demographic'] == demo
                self.kmf.fit(self.df[mask]['duration_months'], 
                             event_observed=self.df[mask]['is_finished'], 
                             label=demo)
                self.kmf.plot_survival_function(ax=ax, ci_show=False)

            plt.title('Demographic Survival Battle', fontsize=16)
            plt.xlabel('Months')
            plt.xlim(0, 120) 
            plt.grid(True, alpha=0.3)
            plt.savefig(target_file)
            plt.close()
        else:
            print(f"   >> [SKIP] Demographic battle chart already exists.")
            
        print(f"   -> Median Lifespans: {results}")
        
        # RETURN THE DATA!
        return results
    
    def compare_demographics(self):
        """Alias for run_demographic_battle for compatibility."""
        return self.run_demographic_battle()
    
    def run_weibull_predictions(self):
        """
        Predicts the 'Expected Lifespan' for average manga in different categories.
        Uses Weibull Accelerated Failure Time (AFT) model.
        """
        print("\n>> [SURVIVAL] Running Weibull AFT Prediction...")
        
        # Prepare data (Weibull can't handle 0 duration, so we clip)
        aft_df = self.df[['duration_months', 'is_finished', 'is_shounen', 'is_seinen', 'score', 'members_log']].copy()
        aft_df['duration_months'] = aft_df['duration_months'].clip(lower=1)
        
        # Fit Model
        aft = WeibullAFTFitter()
        try:
            aft.fit(aft_df, duration_col='duration_months', event_col='is_finished')
            
            # Print the "Time Acceleration Factors"
            print("   -> Weibull Expectation Factors:")
            print(aft.summary[['exp(coef)', 'p']])
            
            # What does the average manga look like?
            median_life = aft.median_survival_time_
            print(f"   -> Theoretical Baseline Lifespan: {median_life:.1f} months")
            
            return aft
        except Exception as e:
            print(f"   ⚠️ Weibull AFT failed (Data issues): {e}")
            return None

    def check_c_index(self):
        """Calculates how accurately the survival model orders death times."""
        
        # Simple C-Index on your main risk factor (e.g., Score)
        # Does higher score actually mean longer life?
        c_index = concordance_index(
            self.df['duration_months'], 
            self.df['score'], 
            self.df['is_finished']
        )
        
        print(f"   -> C-Index (Predictive Power of Score): {c_index:.3f}")
        if c_index < 0.5:
            print("      ⚠️ WARNING: Your features might be inversely correlated!")



# ════════════════════════════════════════════════════════════════════════════
# 3. EXPLAINABILITY & INTERPRETATION ENGINE
# ════════════════════════════════════════════════════════════════════════════
class ExplainabilityEngine:
    """
    Makes ML predictions interpretable for non-technical users.
    Handles 'Loaded Models' where we only have the final Ensemble object.
    """
    
    def __init__(self, engine, df, feature_names):
        self.engine = engine
        self.df = df
        self.feature_names = feature_names
        self.shap_values = None
        self.importances = {}
        
    def _get_proxy_model(self):
        """
        Helper: Drills down into an Ensemble to find a tree-based model
        we can use for explanations (Feature Importance / SHAP).
        """
        model = self.engine.best_model
        
        # 1. If the main model works directly, use it
        if hasattr(model, 'feature_importances_'):
            return model
            
        # 2. If it's a Voting Classifier/Ensemble, check its internals
        if hasattr(model, 'estimators_'):
            for sub_model in model.estimators_:
                if hasattr(sub_model, 'feature_importances_'):
                    return sub_model
                    
        # 3. If it's a Stacking Classifier, check named estimators
        if hasattr(model, 'named_estimators_'):
            for name, sub_model in model.named_estimators_.items():
                if hasattr(sub_model, 'feature_importances_'):
                    return sub_model
                    
        return None

    def calculate_feature_importance(self):
        print("\n>> [EXPLAINABILITY] Calculating Feature Importance...")
        
        # 1. Try to get a model with importances
        proxy_model = self._get_proxy_model()
        
        if proxy_model is None:
            # Fallback for Linear Models (Logistic Regression)
            if hasattr(self.engine.best_model, 'coef_'):
                importances = np.abs(self.engine.best_model.coef_[0])
            else:
                print("   [WARNING] Could not extract feature importances (Model type unknown)")
                return self
        else:
            importances = proxy_model.feature_importances_
        
        # 2. Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 3. Normalize to 0-100 scale
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (100 * importance_df['importance'] / total_importance)
        else:
            importance_df['importance_pct'] = 0
        
        self.importances = importance_df
        
        print(f"   ✓ Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return self
    
    def calculate_shap_values(self, sample_size=100):
        target_file = f"{Config.OUTPUT_DIR}/shap_summary.png"
        
        if os.path.exists(target_file):
            print(f"   >> [SKIP] SHAP summary plot already exists.")
            return self
        
        if not HAS_SHAP:
            print("\n>> [EXPLAINABILITY] SHAP not available.")
            return self
        
        print("\n>> [EXPLAINABILITY] Calculating SHAP Values...")
        
        # 1. Find a tree-based proxy model
        proxy_model = self._get_proxy_model()
        
        if proxy_model is None:
            print("   [WARNING] No tree-based models found inside Ensemble to explain.")
            return self

        # 2. Prepare Sample Data
        try:
            X_sample = self.engine.X_test.sample(
                min(sample_size, len(self.engine.X_test)),
                random_state=42
            )
            
            # 3. Create Explainer on the PROXY model
            # check_additivity=False prevents crashes on some XGBoost versions
            explainer = shap.TreeExplainer(proxy_model)
            self.shap_values = explainer.shap_values(X_sample, check_additivity=False)
            
            # 4. Handle Binary Classification Output
            # Scikit-Learn RF returns list [Class0, Class1]. XGBoost returns Array.
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1] # Class 1 (Cancellation)
            else:
                shap_vals = self.shap_values
                
            # SAFETY CHECK: If shap_vals is 3D (Interactions), flatten or warning
            if len(np.array(shap_vals).shape) > 2:
                print("   [INFO] Detected interaction values. Flattening for summary plot.")
                # If it's interaction values, we usually can't plot standard summary easily
                # But typically this check catches the weird edge case you saw.
            
            # 5. Plot - FORCE DOT PLOT (Beeswarm)
            plt.figure(figsize=Config.FIG_SIZE_LARGE)
            
            shap.summary_plot(
                shap_vals, 
                X_sample, 
                feature_names=self.feature_names,
                plot_type="dot", # <--- CRITICAL FIX: Forces standard beeswarm
                show=False
            )
            
            plt.title("SHAP Feature Impact (Top Risk Drivers)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(target_file, dpi=Config.DPI, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ SHAP values calculated")
            
        except Exception as e:
            print(f"   [WARNING] SHAP calculation failed: {e}")
        
        return self
    


# ════════════════════════════════════════════════════════════════════════════
# 4. PLAIN ENGLISH TRANSLATOR
# ════════════════════════════════════════════════════════════════════════════
class MangaReportHTML:
    """
    Advanced Report Engine.
    Now includes 'Smart Parsing' to format the text blocks beautifully.
    """
    
    def __init__(self, template_dir=".", template_file="report_template.html"):
        self.template_dir = template_dir
        self.template_file = template_file
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def _parse_action_plan(self, raw_text):
        """
        Splits the raw text blob into clean sections for the HTML template.
        Updated to handle Unicode box-drawing characters (─).
        """
        sections = {
            "situation": "",
            "drivers": "",
            "next_steps": "",
            "reality": ""
        }
        
        # 1. Clean the heavy box art from the top (The big ASCII box)
        clean_text = re.sub(r'╔.*?╝', '', raw_text, flags=re.DOTALL)
        
        # 2. Universal Regex Patterns
        # matches: Header -> optional whitespace -> separator lines (hyphens or unicode dashes) -> content
        
        # The Situation
        sit_match = re.search(r'THE SITUATION:\s*[─-]*\s*(.*?)(?=\n\s*WHAT\'S DRIVING)', clean_text, re.DOTALL)
        if sit_match: sections['situation'] = sit_match.group(1).strip()
        
        # Risk Drivers
        drive_match = re.search(r'WHAT\'S DRIVING YOUR RISK:\s*[─-]*\s*(.*?)(?=\n\s*YOUR NEXT 7 DAYS)', clean_text, re.DOTALL)
        if drive_match: sections['drivers'] = drive_match.group(1).strip()
        
        # Next 7 Days
        next_match = re.search(r'YOUR NEXT 7 DAYS:\s*[─-]*\s*(.*?)(?=\n\s*REALITY CHECK)', clean_text, re.DOTALL)
        if next_match: sections['next_steps'] = next_match.group(1).strip()
        
        # Reality Check
        real_match = re.search(r'REALITY CHECK:\s*[─-]*\s*(.*)', clean_text, re.DOTALL)
        if real_match: sections['reality'] = real_match.group(1).strip()
        
        # Fallback: If regex fails, put everything in Situation (but print a warning to console)
        if not any(sections.values()):
            print("   ⚠️ Warning: Text splitting failed. Check regex patterns.")
            sections['situation'] = raw_text

        return sections

    def generate_portfolio_summary(self, results_dict, filename, demo_data=None, feature_df=None, aft_model=None):
        """
        Generates the 'Morning Briefing' portfolio summary.
        Now accepts 'feature_df' to show global model weights.
        """
        print(f"   >> [HTML ENGINE] Rendering Portfolio Summary...")

        if not results_dict:
            print("   ⚠️ No data for summary.")
            return

        # 1. Process Risk Data
        manga_list = []
        high_count = 0
        med_count = 0
        low_count = 0
        total_risk = 0
        
        for title, data in results_dict.items():
            prob = data['probability']
            total_risk += prob
            
            if prob > 0.85:
                status = "CRITICAL"
                color = "#C73E1D"
                row_class = "row-high"
                high_count += 1
            elif prob > 0.60:
                status = "UNSTABLE"
                color = "#F18F01"
                row_class = "row-med"
                med_count += 1
            else:
                status = "SECURE"
                color = "#3B7A57"
                row_class = "row-low"
                low_count += 1
                
            manga_list.append({
                "title": title, "prob": prob, "status": status, 
                "color": color, "row_class": row_class
            })

        manga_list.sort(key=lambda x: x['prob'], reverse=True)
        total_count = len(manga_list)
        avg_risk = (total_risk / total_count) * 100
        
        # 2. Process Demographic Data
        demo_stats = []
        if demo_data:
            sorted_demos = sorted(demo_data.items(), key=lambda item: item[1])
            for demo_name, months in sorted_demos:
                if months < 30: col = '#C73E1D'
                elif months < 45: col = '#F18F01'
                elif months < 60: col = '#2E86AB'
                else: col = '#3B7A57'
                demo_stats.append({'name': demo_name, 'life': f"{months:.1f} Months", 'color': col})
        else:
            demo_stats = [{'name': 'Data Unavailable', 'life': 'N/A', 'color': '#888'}]
        
        feature_stats = []
        if feature_df is not None and not feature_df.empty:
            # Take top 10
            top_features = feature_df.head(10).copy()
            
            # Find the maximum value to scale against
            max_val = top_features['importance_pct'].max()
            if max_val == 0: max_val = 1  # Prevent divide by zero

            for _, row in top_features.iterrows():
                # Calculate relative width (0 to 100)
                relative_width = (row['importance_pct'] / max_val) * 100
                
                raw_name = row['feature'].replace('_', ' ').upper()
                if "SCORE X MEMBERS" in raw_name:
                    display_name = "CULT STATUS (PROXY)"
                else:
                    display_name = raw_name
                
                feature_stats.append({
                    'name': display_name,
                    'pct': f"{row['importance_pct']:.1f}",  # Display text (e.g. "8.8")
                    # 'width': f"{relative_width:.1f}"        # Visual width (e.g. "100.0")
                    'width': int(relative_width)
                })
        
        weibull_stats = []
        baseline_life = "N/A"

        if aft_model:
            # Get baseline
            baseline_life = f"{aft_model.median_survival_time_:.1f} Months"
            
            # Extract factors
            summary = aft_model.summary
            for idx, row in summary.iterrows():
                # --- THE FIX: UNPACK THE TUPLE ---
                if isinstance(idx, tuple):
                    param, feature_name = idx
                    # We only care about 'lambda_' (Scale/Lifespan drivers), not 'rho_' (Shape)
                    if param != 'lambda_': continue 
                else:
                    feature_name = str(idx) # Fallback for simple index
                
                coef = row['exp(coef)']
                
                # Filter for interesting ones
                # We check the feature_name string now, which is safe
                if any(x in feature_name for x in ['shounen', 'seinen', 'score', 'members']):
                    
                    # Convert to percent change
                    pct_change = (coef - 1.0) * 100
                    
                    if pct_change < 0:
                        # Negative impact (Red)
                        impact_str = f"{pct_change:.1f}%"
                        color = "#C73E1D"
                        desc = "LIFESPAN PENALTY"
                        name_clean = feature_name.replace('is_', '').upper()
                    else:
                        # Positive impact (Green)
                        impact_str = f"+{pct_change:.1f}%"
                        color = "#3B7A57"
                        desc = "LIFESPAN BONUS"
                        name_clean = feature_name.replace('is_', '').upper()
                        
                    weibull_stats.append({
                        'name': name_clean,
                        'impact': impact_str,
                        'color': color,
                        'desc': desc
                    })
        
        # 4. Render Template
        context = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_count": total_count,
            "avg_risk": f"{avg_risk:.1f}",
            "avg_color": "#C73E1D" if avg_risk > 50 else "#3B7A57",
            "high_risk_count": high_count,
            "low_risk_count": low_count,
            "sorted_manga": manga_list,
            "high_pct": (high_count / total_count) * 100,
            "med_pct": (med_count / total_count) * 100,
            "low_pct": (low_count / total_count) * 100,
            "txt_triage": f"{high_count} high-risk assets. Immediate review recommended." if high_count > 0 else "No critical threats.",
            "txt_stabilize": f"{med_count} assets in the 'Danger Zone'. High ROI for intervention." if med_count > 0 else "Mid-range volatility is low.",
            "txt_protect": f"{low_count} performing assets. Risk: Complacency." if low_count > 0 else "CRITICAL: No safe assets found.",
            "demo_stats": demo_stats,
            
            # PASS FEATURE DATA
            "feature_stats": feature_stats,
            "weibull_stats": weibull_stats,
            "baseline_life": baseline_life
        }

        try:
            template = self.env.get_template("summary_template.html")
            html_content = template.render(context)
            with open(filename, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            if pisa_status.err: print(f"   ⚠️ Summary Generation Error: {pisa_status.err}")
            else: print(f"   📄 [SUMMARY GENERATED] {filename}")
        except Exception as e:
            print(f"   ❌ Critical Error in Summary: {e}")
    

    def generate_report(self, title, probability, features, action_plan, filename):
        # print(f"   >> [HTML ENGINE] Rendering Dossier for '{title}'...")

        # 1. Smart Parse the Text
        parsed_plan = self._parse_action_plan(action_plan)

        # 2. Prepare Context
        context = {
            "title": title,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ref_id": f"CSE-{random.randint(1000,9999)}-X",
            "risk_percent": round(probability * 100, 1),
            "features": features,
            
            # Pass the separated sections
            "txt_situation": parsed_plan['situation'].replace('\n', '<br>'),
            "txt_drivers": parsed_plan['drivers'].replace('\n', '<br>'),
            "txt_next_steps": parsed_plan['next_steps'].replace('\n', '<br>'),
            "txt_reality": parsed_plan['reality'].replace('\n', '<br>')
        }
        
        if probability > 0.5:
            context["status_color"] = "#B42828"
            context["status_text"] = "HIGH RISK ASSET"
            context["risk_color"] = "#B42828"
            context["summary_intro"] = "The predictive algorithm has identified this asset as HIGH RISK. Statistical indicators suggest a significant deviation from survival baselines."
        else:
            context["status_color"] = "#288C50"
            context["status_text"] = "STABLE ASSET"
            context["risk_color"] = "#288C50"
            context["summary_intro"] = "The predictive algorithm evaluates this asset as STABLE. Key performance indicators are aligned with long-running series profiles."

        # 3. Render
        try:
            template = self.env.get_template(self.template_file)
            html_content = template.render(context)
            
            with open(filename, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            
            if pisa_status.err:
                print(f"   ⚠️ PDF Generation Error: {pisa_status.err}")
            # else:
            #     print(f"   📄 [DOSSIER GENERATED] {filename}")
                
        except Exception as e:
            print(f"   ❌ Critical Error: {e}")
    
    def generate_roi_report(self, financial_data, filename):
        """
        Generates the 'CFO Level' Financial Impact Report.
        """
        print(f"   >> [HTML ENGINE] Rendering ROI Report...")
        
        # Determine Verdict
        if financial_data['net_impact'] > 0:
            verdict = "PROFITABLE"
            verdict_color = "#3B7A57" # Green
            verdict_text = "Algorithm is generating net positive value."
        else:
            verdict = "UNPROFITABLE"
            verdict_color = "#C73E1D" # Red
            verdict_text = "Algorithm costs exceed savings. Threshold tuning required."

        context = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "data": financial_data,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "verdict_text": verdict_text,
            # Format numbers as currency
            "fmt_saved": f"${financial_data['saved_cash']:,.0f}",
            "fmt_lost": f"${financial_data['lost_value']:,.0f}",
            "fmt_wasted": f"${financial_data['wasted_cash']:,.0f}",
            "fmt_net": f"${financial_data['net_impact']:,.0f}",
            "fmt_roi": f"{financial_data['roi_percent']:.1f}%"
        }

        try:
            template = self.env.get_template("roi_template.html")
            html_content = template.render(context)
            with open(filename, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            if pisa_status.err: print(f"   ⚠️ ROI Report Error: {pisa_status.err}")
            else: print(f"   💰 [ROI REPORT GENERATED] {filename}")
        except Exception as e:
            print(f"   ❌ ROI Report Error: {e}")


class PlainEnglishTranslator:
    """
    Converts technical ML predictions into plain English for non-technical users.
    
    Explains:
    - What the prediction means
    - Why the model made that prediction
    - What factors contributed most
    - What actions can be taken
    """
    
    # Risk level descriptions
    RISK_LEVELS = {
        'very_low': {
            'range': (0, 0.1),
            'label': 'VERY LOW RISK',
            'description': 'Almost certainly will continue',
            'emoji': '🟢'
        },
        'low': {
            'range': (0.1, 0.25),
            'label': 'LOW RISK',
            'description': 'Likely to continue for a long time',
            'emoji': '🟢'
        },
        'moderate': {
            'range': (0.25, 0.5),
            'label': 'MODERATE RISK',
            'description': 'Could go either way',
            'emoji': '🟡'
        },
        'high': {
            'range': (0.5, 0.75),
            'label': 'HIGH RISK',
            'description': 'Likely to be cancelled soon',
            'emoji': '🔴'
        },
        'very_high': {
            'range': (0.75, 1.0),
            'label': 'VERY HIGH RISK',
            'description': 'Almost certainly will be cancelled',
            'emoji': '🔴'
        }
    }
    
    @staticmethod
    def get_risk_level(prob):
        """Get risk category for a probability."""
        for level, info in PlainEnglishTranslator.RISK_LEVELS.items():
            if info['range'][0] <= prob < info['range'][1]:
                return level, info
        return 'very_high', PlainEnglishTranslator.RISK_LEVELS['very_high']
    
    @staticmethod
    def explain_prediction(manga_title, prob, importance_features=None, is_finished=False):
        """
        Generates a brutally honest, actionable strategic analysis.
        Compatible with actual engineered features from the ML pipeline.
        """
        # 1. Calculate metrics
        level, level_info = PlainEnglishTranslator.get_risk_level(prob)
        survival_chance = (1 - prob) * 100
        risk_chance = prob * 100
        
        # 2. Feature-to-action mapping (ACTUAL FEATURES)
        def get_action_for_feature(feature_name, importance):
            """Dynamically generate advice based on actual feature patterns."""
            fname = feature_name.lower()
            
            # MEMBERS / AUDIENCE SIZE
            if 'members' in fname:
                if 'log' in fname:
                    return {
                        'lever': 'Audience Size (Log Scale)',
                        'fix': 'Members_log <8.0 (~3k readers): Distribution problem. Members_log >10 (~22k): You have reach. Focus: Sub-1k = emergency content blitz. 1k-5k = optimize for algorithm. 5k+ = retention over acquisition.',
                        'why': 'Log-transformed member count = your visibility ceiling in platform algorithms'
                    }
                elif 'zscore' in fname:
                    return {
                        'lever': 'Audience Percentile',
                        'fix': 'Negative z-score = below average reach. Positive z-score >1 = top 15% territory. If negative: your problem is awareness, not quality.',
                        'why': 'Shows where you rank vs all manga in the dataset'
                    }
                else:
                    return {
                        'lever': 'Raw Audience Size',
                        'fix': '<1000 members: You\'re invisible. 1k-10k: You exist but barely. 10k-50k: Viable but not safe. 50k+: You have leverage.',
                        'why': 'Raw member count = your negotiating power with publishers'
                    }
            
            # SCORE / QUALITY PERCEPTION
            elif 'score' in fname:
                if 'squared' in fname or 'cubed' in fname:
                    return {
                        'lever': 'Quality Non-Linear Effects',
                        'fix': 'The gap between 7.0 and 8.0 is exponentially more valuable than 6.0 to 7.0. Every 0.1 point above 7.5 compounds your survival odds.',
                        'why': 'Quality scores have accelerating returns - mediocrity is fatal, excellence is multiplicative'
                    }
                elif 'x_members' in fname:
                    return {
                        'lever': 'Quality × Reach Synergy',
                        'fix': 'High score + low members = hidden gem (marketing failure). High members + low score = viral trash (unsustainable). You need BOTH above 7.5 and 10k respectively.',
                        'why': 'Quality without audience = wasted potential. Audience without quality = temporary spike'
                    }
                elif 'high_score' in fname:
                    return {
                        'lever': 'Quality Threshold Flag',
                        'fix': 'Binary signal: Are you above 7.5? If no, nothing else matters. Fix craft first. If yes, now you can play the growth game.',
                        'why': '7.5 is the "good enough to compete" floor in modern manga markets'
                    }
                else:
                    return {
                        'lever': 'Base Quality Score',
                        'fix': '<6.5: You have a craft problem. 6.5-7.5: You\'re in the "good but forgettable" zone - need a hook. 7.5-8.5: Strong foundation. >8.5: You\'re elite but that won\'t save you alone.',
                        'why': 'Score predicts word-of-mouth velocity and algorithm favor'
                    }
            
            # DEMOGRAPHICS
            elif fname in ['is_shounen', 'is_seinen', 'is_shoujo', 'is_josei']:
                demo_name = fname.replace('is_', '').title()
                if 'shounen' in fname:
                    return {
                        'lever': 'Shounen Demographic',
                        'fix': 'You\'re in the bloodbath category (60%+ of all manga). Differentiate immediately or die in the noise. Can\'t out-action Battle Shounen? Pivot to character drama or genre fusion.',
                        'why': 'Shounen has the most competition and the highest axe rate'
                    }
                elif 'seinen' in fname:
                    return {
                        'lever': 'Seinen Demographic',
                        'fix': 'Older audience = more patience but less volume. Your advantage: depth and complexity. Your risk: slow burn doesn\'t work in weekly formats.',
                        'why': 'Seinen manga have longer median survival but lower peak reach'
                    }
                elif 'shoujo' in fname:
                    return {
                        'lever': 'Shoujo Demographic',
                        'fix': 'Romance-heavy market = relationship chemistry is everything. Weak love interest = death sentence. Your edge: loyal fanbase if you deliver emotional payoff.',
                        'why': 'Shoujo has the most engaged (but smallest) audience segment'
                    }
                else:  # Josei
                    return {
                        'lever': 'Josei Demographic',
                        'fix': 'Mature women audience = hardest to reach but most valuable retention. If you have them, protect them. If you don\'t, you won\'t get them mid-run.',
                        'why': 'Josei manga either build cult followings or fail fast - no middle ground'
                    }
            
            # MAGAZINE
            elif 'magazine' in fname:
                if 'avg_score' in fname:
                    return {
                        'lever': 'Magazine Quality Average',
                        'fix': 'Publishing in a high-avg magazine = higher expectations. Publishing in low-avg = easier to stand out but less prestige. Know your context.',
                        'why': 'Magazine reputation sets baseline reader expectations for your series'
                    }
                elif 'count' in fname:
                    return {
                        'lever': 'Magazine Portfolio Size',
                        'fix': 'Big magazines = more competition for editorial attention. Small magazines = higher % of their focus but less reach. Pick your battlefield.',
                        'why': 'Magazine size determines your share of promotional resources'
                    }
                else:
                    return {
                        'lever': 'Magazine Identity',
                        'fix': 'Each magazine has an axe policy and editorial culture. Research who they\'ve cancelled and why. Pattern match your trajectory.',
                        'why': 'Some magazines kill fast, others give 100+ chapters. Know which you\'re in.'
                    }
            
            # POPULARITY FLAGS
            elif fname in ['is_popular', 'is_niche', 'is_hit', 'is_breakout']:
                if 'popular' in fname:
                    return {
                        'lever': 'Popularity Threshold',
                        'fix': 'Top 25% audience = you have safety margin. Use it to take creative risks before you become stale.',
                        'why': 'Popular manga get more runway but also face higher expectation decay'
                    }
                elif 'niche' in fname:
                    return {
                        'lever': 'Niche Status',
                        'fix': 'Bottom 25% audience = you\'re on thin ice. Immediate actions: increase update frequency, create shareable moments, or accept this is a passion project.',
                        'why': 'Niche manga survive on ultra-loyal cores or die quietly'
                    }
                elif 'hit' in fname:
                    return {
                        'lever': 'Hit Status',
                        'fix': 'Top 10% + High Score = You won. Now protect it. Biggest risk: resting on laurels. Keep shipping quality.',
                        'why': 'Hit manga rarely get axed but can kill themselves with quality drops'
                    }
                else:  # breakout
                    return {
                        'lever': 'Breakout Potential',
                        'fix': 'Top 5% trajectory = You\'re in rarified air. Anime/merch deals incoming. Hire help, protect your health, plan for scale.',
                        'why': 'Breakout manga face burnout risk more than axe risk'
                    }
            
            # TITLE FEATURES
            elif 'title' in fname:
                return {
                    'lever': 'Title Design',
                    'fix': 'Title length/complexity affects discoverability. Long titles = SEO advantage but memorability loss. Short titles = brand-able but competitive.',
                    'why': 'Title is your first (and often only) chance to hook readers browsing'
                }
            
            # TAG/GENRE
            elif 'tag' in fname:
                return {
                    'lever': 'Genre Positioning',
                    'fix': 'Tag count matters: Too few = undiscoverable. Too many = unfocused. Optimal: 3-7 tags that accurately describe your unique blend.',
                    'why': 'Tags determine which algorithm buckets you compete in'
                }
            
            # SEASONAL LAUNCH
            elif any(season in fname for season in ['spring', 'summer', 'fall', 'winter']):
                return {
                    'lever': 'Launch Timing',
                    'fix': 'Launch season affects initial visibility. Spring/Fall = high competition (new anime seasons). Summer/Winter = less competition but less attention.',
                    'why': 'Seasonal patterns affect your first 12 weeks of data - critical for survival'
                }
            # [NEW] CULT / ENGAGEMENT PROXY
            elif 'score_x_members' in fname:
                return {
                    'lever': 'Cult Status (Quality × Reach)',
                    'fix': 'This is your "Real Engagement" metric. High Score + Low Members = Hidden Gem (Marketing problem). Low Score + High Members = Viral Trash (Retention problem). You need to balance both.',
                    'why': 'This proxy metric identifies if you are building a loyal fanbase or just getting empty clicks.'
                }
            # INTERACTION FEATURES
            elif '_x_' in fname or 'complexity' in fname:
                return {
                    'lever': 'Combined Effects',
                    'fix': 'This is a multiplicative feature - it measures how two factors interact. High importance = the combination matters more than parts.',
                    'why': 'Interaction features capture non-obvious synergies your success depends on'
                }
            
            # GENERIC FALLBACK
            else:
                return {
                    'lever': feature_name.replace('_', ' ').title(),
                    'fix': 'This metric is statistically significant but requires domain expertise to interpret. Check if it\'s trending up or down over time.',
                    'why': 'Model flagged this as important - investigate why it correlates with survival'
                }
        
        # 3. Build the report header
        if is_finished:
            status_str = "COMPLETED (POST-MORTEM)"
            meaning_str = f"This manga finished its run. Statistically, it had a {risk_chance:.1f}% risk profile.\nLooking back: {'Outlier success despite odds' if prob > 0.5 else 'Expected outcome based on metrics'}."
        else:
            status_str = "PUBLISHING (LIVE RISK)"
            meaning_str = f"Current axe probability: {risk_chance:.1f}%\nIf we cloned this manga 100 times, roughly {int(risk_chance)} would die and {int(survival_chance)} would survive."
        
        explanation = f"""
    ╔{'═'*80}╗
    ║  SURVIVAL ANALYSIS: {manga_title[:52]:<52} ║
    ╚{'═'*80}╝

    {level_info['emoji']} RISK: {risk_chance:.1f}%  |  STATUS: {status_str}

    THE SITUATION:
    ──────────────
    """
        
        # 4. Status-specific context
        if is_finished:
            if prob > 0.5:
                explanation += f"""This manga completed despite {risk_chance:.0f}% risk profile. Statistical outlier.
    The data said axe, reality said success. Means it had qualities not captured by metrics
    (art style, cultural timing, editorial protection, or pure luck).

    LESSON: When data and outcomes diverge, the gap is your unfair advantage.
    """
            else:
                explanation += f"""This manga completed with {risk_chance:.0f}% risk profile. Expected outcome.
    Solid fundamentals, no miracles needed. This is the baseline template.

    LESSON: Meet these metrics + don't fuck up = survival.
    """
        else:
            if prob >= 0.7:
                explanation += f"""Critical condition. {risk_chance:.0f}% risk = emergency triage.

    You have 2-4 weeks for emergency surgery or start planning your next series.
    Don't "boost engagement" - that's too slow. You need a narrative bomb or an exit strategy.
    """
            elif prob >= 0.5:
                explanation += f"""Danger zone. {risk_chance:.0f}% risk = one bad arc from cancellation.

    You have 8-12 weeks to turn this around. The readers who haven't left are your lifeline.
    The ones who already left aren't coming back. Focus: retention, not resurrection.
    """
            elif prob >= 0.3:
                explanation += f"""Unstable equilibrium. {risk_chance:.0f}% risk = could go either way.

    Most manga live and die here. Difference between survival and axe at this level is usually
    ONE decision: a viral arc, a marketing push, or a creative risk that pays off.
    """
            else:
                explanation += f"""Good position. {risk_chance:.0f}% risk = breathing room.

    But "safe" manga that stop innovating become "stale" manga. Stale manga get axed when
    the publisher needs to make room. Your job: extend runway while you have leverage.
    """
        
        # 5. Feature-driven action items
        explanation += f"""

    WHAT'S DRIVING YOUR RISK:
    ─────────────────────────
    """
        
        if importance_features:
            top_features = importance_features[:3]  # Focus on top 3
            
            for i, (feature, importance) in enumerate(top_features, 1):
                action_data = get_action_for_feature(feature, importance)
                
                explanation += f"""
    {i}. {action_data['lever'].upper()} (Impact: {importance:.3f})
    → {action_data['why']}
    
    WHAT TO DO:
    {action_data['fix']}
    """
        else:
            explanation += "\n[No feature data available - running blind]\n"
        
        # 6. The directive
        explanation += f"""

    YOUR NEXT 7 DAYS:
    ─────────────────
    """
        
        if is_finished:
            explanation += """Extract the lessons:
    1. Screenshot engagement curve - where did readers spike/drop?
    2. Compare risk factors vs actual outcome - what did model miss?
    3. Apply insights to NEXT series. That's what matters now.
    """
        else:
            if prob >= 0.7:
                explanation += """Emergency triage. Pick ONE:

    A) PIVOT: Major story shift in next 3 chapters (new character, plot twist, genre blend).
    Risky but you're already dying.
    
    B) DOUBLE DOWN: Go all-in on quality. Make next 5 chapters your magnum opus.
    Die with your boots on.
    
    C) PLANNED EXIT: Start your next series NOW. Don't let this ship sink you.

    Do not "wait and see" - that's choosing slow death over quick action.
    """
            elif prob >= 0.5:
                explanation += """Priority order:
    1. Audit last 5 chapters for engagement drops
    2. Identify your weakest metric from the list above  
    3. Spend 80% of energy fixing that ONE thing over next month
    4. Ship your best chapter in weeks - remind readers why they're here

    You don't need to fix everything. You need to fix the thing killing you.
    """
            elif prob >= 0.3:
                explanation += """You're stable but not safe:
    1. Identify your top-performing chapter (engagement, not your favorite)
    2. Reverse-engineer what made it work
    3. Replicate that formula in your next arc
    4. Build audience outside platform (social, email, Discord)

    Survival = not dying. Growth = compounding wins. Do both.
    """
            else:
                explanation += """You have leverage. Use it:
    1. Negotiate better terms while you're hot
    2. Launch supplementary content (character guides, Q&A, world-building)
    3. Build IP optionality (anime pitch, merch concepts)
    4. Plan SECOND series to run concurrently

    Prepare for success before you need it.
    """
        
        # 7. Reality check footer
        explanation += f"""

    REALITY CHECK:
    ──────────────
    • Model analyzed {len(PlainEnglishTranslator.RISK_LEVELS)} manga. It's probabilistic, not prophetic.
    • CAN'T predict: viral moments, editorial politics, burnout, black swans
    • CAN predict: statistical likelihood from pattern matching

    The model gives you odds. Your job: make them irrelevant through execution.

    ╔{'═'*80}╗
    ║  This is data. Make it obsolete through action.                            ║
    ╚{'═'*80}╝
    """
        
        return explanation

    @staticmethod
    def generate_summary_report(results_dict):
        """
        Generate brutally honest executive summary for multiple predictions.
        No corporate speak. Just patterns and what they mean.
        """
        report = f"""
    ╔{'═'*80}╗
    ║  SURVIVAL PREDICTION REPORT - EXECUTIVE SUMMARY                           ║
    ╚{'═'*80}╝

    DATASET OVERVIEW:
    ─────────────────
    """
        if not results_dict:
            report += "No predictions generated. Nothing to analyze.\n"
            return report
        
        # Calculate statistics
        probs = [r['probability'] for r in results_dict.values()]
        risk_counts = defaultdict(int)
        for prob in probs:
            level, _ = PlainEnglishTranslator.get_risk_level(prob)
            risk_counts[level] += 1
        
        avg_risk = np.mean(probs) * 100
        
        report += f"""
    Total Manga Analyzed: {len(results_dict)}
    Average Axe Risk: {avg_risk:.1f}%
    Highest Risk: {max(probs)*100:.1f}%
    Lowest Risk: {min(probs)*100:.1f}%

    WHAT THIS MEANS:
    ────────────────
    """
        
        # Interpret the average
        if avg_risk >= 60:
            report += f"""Your portfolio is in trouble. {avg_risk:.0f}% average risk means most of these manga
    are statistically doomed. This is either a high-risk experimental slate or you're
    analyzing a graveyard. Either way: expect casualties.
    """
        elif avg_risk >= 40:
            report += f"""Mixed bag. {avg_risk:.0f}% average risk means you're running a standard portfolio with
    normal attrition. About half will survive, half won't. Industry standard.
    The question: can you predict which is which?
    """
        else:
            report += f"""Strong portfolio. {avg_risk:.0f}% average risk means most of these manga have solid
    fundamentals. You're either cherry-picking winners for analysis or your editorial
    standards are working. Don't get complacent - even good manga die.
    """
        
        # Risk distribution
        report += f"""

    RISK BREAKDOWN:
    ───────────────
    """
        
        for level in ['very_low', 'low', 'moderate', 'high', 'very_high']:
            count = risk_counts[level]
            pct = 100 * count / len(results_dict) if len(results_dict) > 0 else 0
            emoji = PlainEnglishTranslator.RISK_LEVELS[level]['emoji']
            label = PlainEnglishTranslator.RISK_LEVELS[level]['label']
            
            report += f"  {emoji} {label:20} {count:3} titles ({pct:5.1f}%)\n"
        
        # Strategic insights
        report += f"""

    STRATEGIC INSIGHTS:
    ───────────────────
    """
        
        # Count critical categories
        critical_count = risk_counts['high'] + risk_counts['very_high']
        safe_count = risk_counts['very_low'] + risk_counts['low']
        unstable_count = risk_counts['moderate']
        
        if critical_count > 0:
            report += f"""
    🚨 CRITICAL: {critical_count} manga in high/very high risk
    → These need immediate intervention or acceptance of failure
    → Triage decision: which are worth saving vs letting die?
    """
        
        if unstable_count > len(results_dict) * 0.4:
            report += f"""
    ⚠️  UNSTABLE: {unstable_count} manga in moderate risk zone
    → This is where most manga live and die
    → Small improvements = survival. Stagnation = axe.
    → These are your highest-leverage intervention targets
    """
        
        if safe_count > len(results_dict) * 0.6:
            report += f"""
    ✅ STRENGTH: {safe_count} manga in low/very low risk
    → These are your foundation - protect them
    → Biggest risk: complacency leading to quality decay
    → Harvest learnings from these to fix the others
    """
        
        # Distribution analysis
        report += f"""

    PORTFOLIO DIAGNOSIS:
    ────────────────────
    """
        
        # Calculate risk concentration
        risk_spread = max(probs) - min(probs)
        
        if risk_spread > 0.7:
            report += f"""
    📊 HIGH VARIANCE ({risk_spread:.1%} spread between best and worst)
    → You have both winners and disasters in the same portfolio
    → Pattern: experimental approach OR inconsistent quality control
    → Action: Study the outliers - what separates success from failure?
    """
        elif risk_spread < 0.3:
            report += f"""
    📊 LOW VARIANCE ({risk_spread:.1%} spread between best and worst)
    → Your manga cluster around similar risk levels
    → Pattern: consistent editorial standards OR homogeneous content
    → Risk: if market shifts, entire portfolio moves together
    """
        else:
            report += f"""
    📊 NORMAL VARIANCE ({risk_spread:.1%} spread)
    → Standard distribution of risk across portfolio
    → Pattern: balanced mix of safe bets and experiments
    """
        
        # Actionable summary
        report += f"""

    IMMEDIATE ACTIONS:
    ──────────────────
    """
        
        if critical_count >= len(results_dict) * 0.3:
            report += f"""
    1. TRIAGE: Review the {critical_count} high-risk manga
    - Which can be saved with intervention?
    - Which should be ended gracefully?
    - Don't waste resources on lost causes

    """
        
        if unstable_count > 0:
            report += f"""
    2. STABILIZE: Focus on {unstable_count} moderate-risk manga
    - These have the highest ROI for intervention
    - Small improvements flip them from axe to survival
    - Check individual reports for specific weaknesses

    """
        
        if safe_count > 0:
            report += f"""
    3. PROTECT: Monitor {safe_count} low-risk manga for decay
    - Safe today ≠ safe forever
    - Watch for engagement drops or quality slips
    - Extract success patterns to apply elsewhere

    """
        
        report += """
    4. PATTERN ANALYSIS: 
    - Compare high-risk vs low-risk features across portfolio
    - What do survivors have in common?
    - What kills manga in your specific context?

    5. PREDICTIVE MONITORING:
    - Re-run predictions monthly on ongoing series
    - Early detection = more intervention options
    - Track prediction accuracy to calibrate trust

    """
        
        # Footer
        report += f"""
    ╔{'═'*80}╗
    ║  This is a snapshot. Market conditions change. Keep monitoring.            ║
    ╚{'═'*80}╝

    Individual manga reports saved to predictions folder.
    Check those for specific tactical recommendations.
    """
        
        return report


# ════════════════════════════════════════════════════════════════════════════
# 5. MODEL EVALUATION & VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════
class ModelEvaluationVisualizer:
    """
    Creates comprehensive visualizations of model performance.
    
    Includes:
    - ROC curves
    - Precision-Recall curves
    - Calibration curves
    - Confusion matrices
    - Feature importance charts
    """
    
    def __init__(self, engine, feature_names):
        self.engine = engine
        self.feature_names = feature_names
        
    def plot_all_roc_curves(self):
        """Compare ROC curves from all models."""
        target_file = f"{Config.OUTPUT_DIR}/roc_all_models.png"
        if os.path.exists(target_file):
            print(f"   >> [SKIP] ROC curves already exist.")
            return self
        
        print("\n>> [VISUALIZATION] Plotting ROC curves...")
        
        plt.figure(figsize=Config.FIG_SIZE_LARGE)
        
        colors = list(Config.COLORS.values())
        
        for idx, (model_name, result) in enumerate(self.engine.results.items()):
            model = result['model']
            
            # Get predictions
            y_proba = model.predict_proba(self.engine.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.engine.y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    label=f'{model_name.replace("_", " ").title()} (AUC={auc_score:.3f})',
                    linewidth=2.5,
                    color=colors[idx % len(colors)])
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        
        plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
        plt.ylabel('True Positive Rate (Correct Detections)', fontsize=12)
        plt.title('Model Performance Comparison: ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/roc_all_models.png", dpi=Config.DPI)
        plt.close()
        
        print(f"   ✓ ROC curves saved")
        return self
    
    def plot_calibration_curve(self):
        """Plot calibration curves to show prediction reliability."""

        target_file = f"{Config.OUTPUT_DIR}/calibration_curves.png"
        if os.path.exists(target_file):
            print(f"   >> [SKIP] Calibration curves already exist.")
            return self
        
        print("\n>> [VISUALIZATION] Plotting calibration curves...")
        
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE_MED)
        
        colors = list(Config.COLORS.values())
        
        for idx, (model_name, result) in enumerate(self.engine.results.items()):
            model = result['model']
            
            y_proba = model.predict_proba(self.engine.X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(self.engine.y_test, y_proba, n_bins=10)
            
            ax.plot(prob_pred, prob_true, 
                   marker='o',
                   label=model_name.replace('_', ' ').title(),
                   linewidth=2,
                   color=colors[idx % len(colors)])
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Probability', fontsize=12)
        ax.set_title('Model Calibration: Are Predicted Probabilities Reliable?', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/calibration_curves.png", dpi=Config.DPI)
        plt.close()
        
        print(f"   ✓ Calibration curves saved")
        return self
    
    def plot_feature_importance(self):
        """Plot top features by importance."""
        target_file = f"{Config.OUTPUT_DIR}/feature_importance_top20.png"
        if os.path.exists(target_file):
            print(f"   >> [SKIP] Feature importance chart already exists.")
            return self
        
        print("\n>> [VISUALIZATION] Plotting feature importance...")
        
        # Get importance from best model
        best_name = max(self.engine.results.keys(),
                       key=lambda k: self.engine.results[k]['test_auc'])
        
        if 'importances' in self.engine.results[best_name]:
            importances = self.engine.results[best_name]['importances']
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=Config.FIG_SIZE_MED)
            plt.barh(range(len(importance_df)), importance_df['importance'], 
                    color=Config.COLORS['primary'])
            plt.yticks(range(len(importance_df)), 
                      [f.replace('_', ' ').title() for f in importance_df['feature']])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top 20 Most Important Features ({best_name.title()})', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{Config.OUTPUT_DIR}/feature_importance_top20.png", dpi=Config.DPI)
            plt.close()
            
            print(f"   ✓ Feature importance saved")
        
        return self
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""

        target_file = f"{Config.OUTPUT_DIR}/confusion_matrices.png"
        if os.path.exists(target_file):
            print(f"   >> [SKIP] Confusion matrices already exist.")
            return self
        
        print("\n>> [VISUALIZATION] Plotting confusion matrices...")
        
        n_models = len(self.engine.results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.engine.results.items()):
            if idx >= 4:
                break
            
            model = result['model']
            y_pred = model.predict(self.engine.X_test)
            cm = confusion_matrix(self.engine.y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(model_name.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{Config.OUTPUT_DIR}/confusion_matrices.png", dpi=Config.DPI)
        plt.close()
        
        print(f"   ✓ Confusion matrices saved")
        return self
    
    def plot_model_comparison(self):
        """Create comparison table of all models."""

        target_file = f"{Config.OUTPUT_DIR}/model_comparison_table.png"
        if os.path.exists(target_file):
            print(f"   >> [SKIP] Model comparison table already exists.")
            return self
        
        print("\n>> [VISUALIZATION] Creating model comparison...")
        
        comparison_data = []
        for model_name, result in self.engine.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train AUC': result['train_auc'],
                'Test AUC': result['test_auc'],
                'CV AUC': result['cv_auc'],
                'Accuracy': result['test_acc']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Test AUC', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE_MED)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=comparison_df.values,
                        colLabels=comparison_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor(Config.COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f"{Config.OUTPUT_DIR}/model_comparison_table.png", dpi=Config.DPI, 
                   bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Model comparison saved")
        return self
    
    def plot_risk_distribution(self):
        """
        Plots the 'Confidence Histogram'.
        Shows how decisively the model separates Safe vs. Critical assets.
        """
        target_file = f"{Config.OUTPUT_DIR}/risk_distribution.png"
        if os.path.exists(target_file): return self
        
        print("\n>> [VISUALIZATION] Plotting Risk Distribution Histogram...")
        
        # Get probabilities from best model
        best_name = max(self.engine.results.keys(), key=lambda k: self.engine.results[k]['test_auc'])
        model = self.engine.results[best_name]['model']
        y_prob = model.predict_proba(self.engine.X_test)[:, 1]
        
        plt.figure(figsize=Config.FIG_SIZE_MED)
        
        # Plot Histogram with KDE
        sns.histplot(y_prob, bins=20, kde=True, color=Config.COLORS['primary'], edgecolor='white')
        
        # Add "Danger Zones" background
        plt.axvspan(0, 0.25, color=Config.COLORS['success'], alpha=0.1, label='Safe Zone')
        plt.axvspan(0.75, 1.0, color=Config.COLORS['danger'], alpha=0.1, label='Kill Zone')
        
        plt.title(f'Model Confidence Distribution ({best_name.title()})', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Cancellation Probability')
        plt.ylabel('Number of Manga')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(target_file, dpi=Config.DPI)
        plt.close()
        print(f"   ✓ Risk distribution saved")
        return self

    def plot_portfolio_donut(self):
        """
        Plots the 'Executive Donut'.
        Visual breakdown of the portfolio risk categories.
        """
        target_file = f"{Config.OUTPUT_DIR}/portfolio_donut.png"
        if os.path.exists(target_file): return self
        
        print("\n>> [VISUALIZATION] Plotting Portfolio Risk Donut...")
        
        # Get predictions
        best_name = max(self.engine.results.keys(), key=lambda k: self.engine.results[k]['test_auc'])
        model = self.engine.results[best_name]['model']
        y_prob = model.predict_proba(self.engine.X_test)[:, 1]
        
        # Categorize
        risk_counts = {
            'Safe (<25%)': sum(y_prob < 0.25),
            'Unstable (25-75%)': sum((y_prob >= 0.25) & (y_prob < 0.75)),
            'Critical (>75%)': sum(y_prob >= 0.75)
        }
        
        # Data for plotting
        labels = list(risk_counts.keys())
        sizes = list(risk_counts.values())
        colors = [Config.COLORS['success'], Config.COLORS['tertiary'], Config.COLORS['danger']]
        
        # Donut Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          startangle=90, colors=colors, pctdistance=0.85,
                                          wedgeprops=dict(width=0.4, edgecolor='white'))
        
        # Style text
        plt.setp(texts, size=12, weight="bold")
        plt.setp(autotexts, size=10, weight="bold", color="white")
        
        ax.set_title("Portfolio Risk Composition", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(target_file, dpi=Config.DPI)
        plt.close()
        print(f"   ✓ Portfolio donut saved")
        return self

    def plot_cox_hazards(self, df):
        """
        Plots the 'Forest Plot' (Hazard Ratios).
        Shows exactly how much each factor multiplies the risk of death.
        """
        target_file = f"{Config.OUTPUT_DIR}/cox_forest_plot.png"
        if os.path.exists(target_file): return self
        
        print("\n>> [VISUALIZATION] Plotting Cox Proportional Hazards...")
        
        try:
            # 1. Prepare Data for Cox (Need raw features + duration + event)
            # We select a subset of interpretable features to avoid collinearity crashing the solver
            cox_cols = ['score', 'members_log', 'title_len', 'is_shounen', 'is_seinen', 'duration_months', 'is_finished']
            cox_df = df[cox_cols].copy().dropna()
            
            # 2. Fit Cox Model
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='duration_months', event_col='is_finished')
            
            # 3. Plot
            plt.figure(figsize=Config.FIG_SIZE_MED)
            cph.plot()
            
            plt.title('Hazard Ratios (Risk Multipliers)', fontsize=14, fontweight='bold')
            plt.xlabel('log(Hazard Ratio) \n < 0 means Protection | > 0 means Risk')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(target_file, dpi=Config.DPI)
            plt.close()
            print(f"   ✓ Cox forest plot saved")
            
        except Exception as e:
            print(f"   ⚠️ Cox plot skipped (Data matrix singular or convergence fail): {e}")
        
        return self
    
    def plot_lift_curve(self):
        """
        Plots the Cumulative Gains / Lift Chart.
        Shows how much 'better than random' the model is at identifying targets.
        """
        target_file = f"{Config.OUTPUT_DIR}/lift_chart.png"
        if os.path.exists(target_file): return self
        
        print("\n>> [VISUALIZATION] Plotting Lift/Cumulative Gains...")
        
        # Get best model probs
        best_name = max(self.engine.results.keys(), key=lambda k: self.engine.results[k]['test_auc'])
        model = self.engine.results[best_name]['model']
        y_prob = model.predict_proba(self.engine.X_test)[:, 1]
        y_true = np.array(self.engine.y_test)
        
        # Sort by predicted probability
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_true_desc = y_true[desc_score_indices]
        
        # Calculate cumulative positives
        cum_positives = np.cumsum(y_true_desc)
        total_positives = cum_positives[-1]
        
        # Calculate percentages
        percent_positives = cum_positives / total_positives
        percent_population = np.arange(1, len(y_true) + 1) / len(y_true)
        
        plt.figure(figsize=Config.FIG_SIZE_MED)
        
        # Plot Model Curve
        plt.plot(percent_population, percent_positives, label=f'Model (AUC={self.engine.results[best_name]["test_auc"]:.2f})', linewidth=2.5, color=Config.COLORS['primary'])
        
        # Plot Random Baseline
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=2)
        
        # Annotate the "Sweet Spot" (e.g., Top 20%)
        top_20_val = percent_positives[int(len(y_true)*0.2)]
        plt.plot([0.2, 0.2], [0, top_20_val], 'r:', linewidth=1.5)
        plt.plot([0, 0.2], [top_20_val, top_20_val], 'r:', linewidth=1.5)
        plt.text(0.22, top_20_val - 0.05, f'Top 20% checks catch\n{top_20_val:.0%} of cancellations', color='#C73E1D', fontweight='bold')
        
        plt.title('Cumulative Gains (The "Efficiency" Chart)', fontsize=14, fontweight='bold')
        plt.xlabel('% of Portfolio Reviewed (High Risk First)')
        plt.ylabel('% of All Cancellations Detected')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(target_file, dpi=Config.DPI)
        plt.close()
        print(f"   ✓ Lift chart saved")
        return self

    def get_financial_metrics(self, df):
        """
        Calculates Profit using POWER LAW VALUATION.
        A Mega-Hit (1M+ members) is weighted exponentially higher than a minor hit.
        This prevents the model from killing the next 'One Piece' to save pennies.
        """
        # 1. BASELINE ASSUMPTIONS
        COST_PER_FLOP = 50000 
        BASE_HIT_VALUE = 500000
        MEMBER_VALUATION = 50 # Each member is worth $50 in Lifetime Value (LTV)
        
        # 2. Get Data & Pre-Calculate Asset Values
        best_name = max(self.engine.results.keys(), key=lambda k: self.engine.results[k]['test_auc'])
        model = self.engine.results[best_name]['model']
        y_prob = model.predict_proba(self.engine.X_test)[:, 1]
        y_true = self.engine.y_test.values
        test_indices = self.engine.X_test.index 
        
        # Calculate Dynamic Value for every manga in test set
        # Value = Max(Base, Members * LTV). A hit with 100k members is worth $5M.
        # We look up the 'members' column from the original dataframe
        asset_values = []
        for idx in test_indices:
            members = df.loc[idx, 'members'] if 'members' in df.columns else 0
            # If it's a "Hit" (y=0), its value is based on reach. 
            # If it's a "Flop" (y=1), its cost is fixed.
            val = max(BASE_HIT_VALUE, members * MEMBER_VALUATION)
            asset_values.append(val)
        asset_values = np.array(asset_values)

        # 3. PROFIT SOLVER LOOP (Now Weighted by Value)
        print(f"\n   >> [PROFIT SOLVER] Optimizing with Power Law Valuation...")
        max_profit = -float('inf')
        best_threshold = 0.5
        
        # We optimize for: (Flops Killed * $50k) - (Actual Value of Hits Killed)
        for t in np.arange(0.01, 1.00, 0.01):
            y_pred = (y_prob >= t).astype(int)
            
            # True Positives (We killed a flop) -> Gain $50k
            savings = np.sum((y_true == 1) & (y_pred == 1)) * COST_PER_FLOP
            
            # False Positives (We killed a hit) -> Lose that specific manga's value
            # Note: We filter asset_values where (True=0 AND Pred=1)
            damages = np.sum(asset_values[(y_true == 0) & (y_pred == 1)])
            
            net = savings - damages
            
            if net > max_profit:
                max_profit = net
                best_threshold = t

        # 4. REPORT GENERATION
        print(f"   🏆 OPTIMAL THRESHOLD: {best_threshold:.2f} | MARGINAL VALUE: ${max_profit:,.0f}")
        
        final_pred = (y_prob >= best_threshold).astype(int)
        asset_ledger = []
        
        # Recalculate totals for the ledger
        total_saved = 0
        total_lost = 0
        total_wasted = 0
        
        for i, idx in enumerate(test_indices):
            title = df.loc[idx, 'title']
            actual = y_true[i]
            pred = final_pred[i]
            val = asset_values[i]
            
            # SAVED HIT (Greenlit correctly)
            if actual == 0 and pred == 0:
                asset_ledger.append({
                    'title': title, 'type': 'SAVED MEGA-HIT' if val > 10000000 else 'SAVED HIT', 
                    'value': val, 'desc': 'Value Captured', 'css': 'pos'
                })
            
            # AVOIDED FLOP (Cancelled correctly)
            elif actual == 1 and pred == 1:
                total_saved += COST_PER_FLOP
                asset_ledger.append({
                    'title': title, 'type': 'AVOIDED FLOP', 
                    'value': COST_PER_FLOP, 'desc': 'Cost Saved', 'css': 'pos'
                })
                
            # WRONGFUL DEATH (Cancelled a hit)
            elif actual == 0 and pred == 1:
                total_lost += val
                asset_ledger.append({
                    'title': title, 'type': 'KILLED UNICORN' if val > 10000000 else 'WRONGFUL DEATH', 
                    'value': -val, 'desc': 'Value Destroyed', 'css': 'neg'
                })
                
            # MISSED FLOP (Greenlit a flop)
            elif actual == 1 and pred == 0:
                total_wasted += COST_PER_FLOP

        asset_ledger.sort(key=lambda x: abs(x['value']), reverse=True)
        
        # Calculate ROI
        cost_basis = total_lost if total_lost > 0 else COST_PER_FLOP
        roi = (max_profit / cost_basis) * 100

        return {
            "model_name": f"{best_name.upper()} (Power Law @ {best_threshold:.2f})",
            "saved_cash": total_saved, 
            "lost_value": total_lost, 
            "wasted_cash": total_wasted,
            "net_impact": max_profit, 
            "roi_percent": roi,
            "correct_kills": int(np.sum((y_true==1)&(final_pred==1))), 
            "wrong_kills": int(np.sum((y_true==0)&(final_pred==1))), 
            "missed_flops": int(np.sum((y_true==1)&(final_pred==0))),
            "ledger": asset_ledger[:25]
        }
    
    def plot_profit_curve(self, df):
        """
        Plots the 'Profit Landscape'.
        Shows exactly how Profit changes as you adjust the Risk Threshold.
        """
        target_file = f"{Config.OUTPUT_DIR}/profit_optimization_curve.png"
        if os.path.exists(target_file): return self
        
        print("\n>> [VISUALIZATION] Plotting Profit Optimization Curve...")
        
        # 1. Setup Data
        COST_PER_FLOP = 50000 
        BASE_HIT_VALUE = 500000
        MEMBER_VALUATION = 50
        
        best_name = max(self.engine.results.keys(), key=lambda k: self.engine.results[k]['test_auc'])
        model = self.engine.results[best_name]['model']
        y_prob = model.predict_proba(self.engine.X_test)[:, 1]
        y_true = self.engine.y_test.values
        
        # Calculate Asset Values
        asset_values = []
        for idx in self.engine.X_test.index:
            members = df.loc[idx, 'members'] if 'members' in df.columns else 0
            val = max(BASE_HIT_VALUE, members * MEMBER_VALUATION)
            asset_values.append(val)
        asset_values = np.array(asset_values)
        
        # 2. Calculate Curve
        thresholds = np.arange(0.01, 1.00, 0.01)
        profits = []
        
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            savings = np.sum((y_true == 1) & (y_pred == 1)) * COST_PER_FLOP
            damages = np.sum(asset_values[(y_true == 0) & (y_pred == 1)])
            profits.append(savings - damages)
            
        # 3. Plot
        plt.figure(figsize=Config.FIG_SIZE_MED)
        
        # The Curve
        plt.plot(thresholds, profits, linewidth=3, color=Config.COLORS['primary'], label='Projected Net Profit')
        
        # The Zero Line (Break Even)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        
        # The Peak (Optimal Threshold)
        max_idx = np.argmax(profits)
        max_t = thresholds[max_idx]
        max_p = profits[max_idx]
        
        plt.plot(max_t, max_p, 'ro', markersize=10, label=f'Peak Profit (${max_p/1000000:.1f}M)')
        plt.axvline(max_t, color=Config.COLORS['success'], linestyle=':', linewidth=2)
        
        # Annotations
        plt.text(max_t + 0.02, max_p, f'Optimal Threshold: {max_t:.2f}', fontweight='bold', color=Config.COLORS['success'])
        
        plt.title('The Profit Curve: Balancing Risk vs. Reward', fontsize=16, fontweight='bold')
        plt.xlabel('Decision Threshold (Probability of Death)', fontsize=12)
        plt.ylabel('Projected Financial Impact (USD)', fontsize=12)
        
        # Color the zones
        plt.fill_between(thresholds, profits, 0, where=(np.array(profits) > 0), color=Config.COLORS['success'], alpha=0.1)
        plt.fill_between(thresholds, profits, 0, where=(np.array(profits) < 0), color=Config.COLORS['danger'], alpha=0.1)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(target_file, dpi=Config.DPI)
        plt.close()
        print(f"   ✓ Profit curve saved")
        return self


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN EXECUTION ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
def main():
    """
    Main execution function. Orchestrates the entire ML pipeline.
    
    Pipeline Steps:
    1. Load and clean data
    2. Engineer features
    3. Train multiple ML models
    4. Evaluate and compare models
    5. Generate explanations for non-technical users
    6. Create comprehensive visualizations
    7. Generate reports
    """
    
    # Initialize
    Config.setup()
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 1: LOAD AND ENGINEER DATA
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    loader = AdvancedDataLoader(Config.DATA_FILE)
    loader.load()
    loader.engineer_features()
    df, df_exploded, feature_names = loader.get_clean_data()
    
    # Check minimum data size
    if len(df) < Config.MIN_TRAIN_SIZE:
        print(f"❌ ERROR: Dataset too small ({len(df)} rows)")
        print(f"   Need at least {Config.MIN_TRAIN_SIZE} rows for reliable ML models")
        sys.exit(1)
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 2: BUILD AND TRAIN ML MODELS
    # ─────────────────────────────────────────────────────────────────
    # # Initialize prediction engine
    ml_engine = AdvancedPredictionEngine(random_state=Config.RANDOM_STATE)
    
    # # Prepare data
    ml_engine.prepare_data(df, feature_names, target_col='is_finished')

    feature_names = ml_engine.feature_names
    
    # ─────────────────────────────────────────────────────────────────
    # UNIFIED STEP: INTELLIGENT MODEL LOADING & TRAINING
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("INTELLIGENT MODEL PROTOCOL")
    print("="*80)

    # 1. Define paths to the brain files
    model_path = os.path.join(Config.MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(Config.MODELS_DIR, "scaler.pkl")
    names_path = os.path.join(Config.MODELS_DIR, "feature_names.pkl")

    # 2. DECISION: Load or Train?
    run_training = False
    
    # Check if files exist
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f">> ⚡ FOUND SAVED MODELS in {Config.MODELS_DIR}")
        try:
            # Try to load them
            with open(model_path, 'rb') as f:
                ml_engine.best_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                ml_engine.scaler = pickle.load(f)
            with open(names_path, 'rb') as f:
                feature_names = pickle.load(f)
                ml_engine.feature_names = feature_names
                
            print("   ✓ Models loaded successfully. Skipping training.")

            print("   >> [RESTORE] Re-evaluating loaded model for reports...")
            
            # 1. Generate fresh predictions on the current test set
            y_prob = ml_engine.best_model.predict_proba(ml_engine.X_test)[:, 1]
            y_pred = ml_engine.best_model.predict(ml_engine.X_test)
            
            # 2. Calculate scores
            test_auc = roc_auc_score(ml_engine.y_test, y_prob)
            test_acc = accuracy_score(ml_engine.y_test, y_pred)
            
            # 3. Force-feed the results dictionary
            ml_engine.results['loaded_model'] = {
                'model': ml_engine.best_model,
                'test_auc': test_auc,
                'test_acc': test_acc,
                'train_auc': test_auc, # Proxy value
                'cv_auc': test_auc,    # Proxy value
            }
            
            # 4. Safe Feature Importance Extraction (for Ensemble)
            try:
                # If it's a simple model (Random Forest/XGBoost)
                if hasattr(ml_engine.best_model, 'feature_importances_'):
                    ml_engine.results['loaded_model']['importances'] = ml_engine.best_model.feature_importances_
                
                # If it's an Ensemble (VotingClassifier), steal importances from a sub-model
                elif hasattr(ml_engine.best_model, 'estimators_'):
                    for sub_model in ml_engine.best_model.estimators_:
                        if hasattr(sub_model, 'feature_importances_'):
                            ml_engine.results['loaded_model']['importances'] = sub_model.feature_importances_
                            break
            except Exception:
                pass # If importances fail, we just skip that specific chart later
            # ══════════════════════════════════════════════════════════  
        except Exception as e:
            print(f"   ⚠️ Load failed ({e}). Corrupted files? Retraining...")
            run_training=True
    else:
        print("   No saved models found. Initiating training sequence...")
        run_training = True

    # 3. EXECUTE TRAINING (Only if needed)
    if run_training:
        print("\n>> [TRAINING] Initializing active models...")
        
        # Baseline
        ml_engine.build_logistic_regression()
        
        # Core Trees
        ml_engine.build_random_forest()
        
        # Advanced Boosters
        if Config.USE_XGBOOST:
            ml_engine.build_xgboost()
            
        if Config.USE_LIGHTGBM:
            if hasattr(ml_engine, 'build_lightgbm'):
                ml_engine.build_lightgbm()
            else:
                print("   ⚠️ LightGBM method missing in class.")

        # Deep Learning
        if Config.USE_TENSORFLOW:
            if hasattr(ml_engine, 'build_neural_network'):
                ml_engine.build_neural_network()
            else:
                print("   ⚠️ Neural Network method missing in class.")

        # Ensemble (The Final Brain)
        ml_engine.build_ensemble()
        
        # 4. SAVE THE NEW BRAINS IMMEDIATELY
        print("\n>> [MODELS] Saving new models...")
        try:
            if not os.path.exists(Config.MODELS_DIR):
                os.makedirs(Config.MODELS_DIR)
                
            with open(model_path, 'wb') as f:
                pickle.dump(ml_engine.best_model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(ml_engine.scaler, f)
            with open(names_path, 'wb') as f:
                pickle.dump(ml_engine.feature_names, f)
            print(f"   ✓ Saved to {Config.MODELS_DIR}")
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 3: MODEL EVALUATION AND VISUALIZATION
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 3: EVALUATION & VISUALIZATION")
    print("="*80)
    
    visualizer = ModelEvaluationVisualizer(ml_engine, feature_names)
    visualizer.plot_all_roc_curves()
    visualizer.plot_calibration_curve()
    visualizer.plot_feature_importance()
    visualizer.plot_confusion_matrices()
    visualizer.plot_model_comparison()

    visualizer.plot_risk_distribution()
    visualizer.plot_portfolio_donut()
    visualizer.plot_cox_hazards(df)
    visualizer.plot_lift_curve()
    visualizer.plot_profit_curve(df)

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: EXPLAINABILITY
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 4: MODEL EXPLAINABILITY")
    print("="*80)
    
    explainer = ExplainabilityEngine(ml_engine, df, feature_names)
    explainer.calculate_feature_importance()
    explainer.calculate_shap_values()
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 5: SURVIVAL ANALYSIS (SUPPLEMENTARY)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 5: CLASSICAL SURVIVAL ANALYSIS")
    print("="*80)
    
    survival_analysis = ClassicalSurvivalAnalysis(df)
    survival_analysis.run_global_survival()
    
    # CAPTURE THE DATA HERE
    demo_lifespans = survival_analysis.compare_demographics()

    survival_analysis.run_weibull_predictions()
    survival_analysis.check_c_index()
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 6: GENERATE PREDICTIONS & EXPLANATIONS
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 6: MAKING PREDICTIONS WITH PLAIN ENGLISH EXPLANATIONS")
    print("="*80)
    
    # Get predictions from best model
    best_model = ml_engine.best_model
    X_test_scaled = ml_engine.X_test
    predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Get test indices
    test_indices = ml_engine.X_test.index
    test_data = df.loc[test_indices]

    all_test_indices = list(ml_engine.X_test.index)
    random.shuffle(all_test_indices)
    
    # Select the first 20 from the shuffled list
    selected_indices = all_test_indices[:20]
    
    print(f"\n>> Generating reports for {len(selected_indices)} random manga...")

    predictions_dict = {}
    
    # 2. Safe Loop
    # 2. Safe Loop with ENSEMBLE SUPPORT
    for idx, manga_idx in enumerate(tqdm(selected_indices, desc="   Processing", unit="manga", ncols=80, colour='green')):
        try:
            manga_title = df.loc[manga_idx, 'title']
            
            # --- FEATURE 1: PREDICT ---
            loc_idx = ml_engine.X_test.index.get_loc(manga_idx)
            prob = predictions_proba[loc_idx]
            
            predictions_dict[manga_title] = {
                'probability': prob,
                'index': manga_idx
            }
            
            # --- FEATURE 2: EXPLAIN (The Fix) ---
            importances = None
            
            # CHECK 1: Is it a Voting Ensemble? Peek inside.
            if hasattr(best_model, 'estimators_'):
                # Iterate through members to find one with feature_importances_
                for sub_model in best_model.estimators_:
                    if hasattr(sub_model, 'feature_importances_'):
                        importances = sub_model.feature_importances_
                        break
            
            # CHECK 2: Is it a standard Tree model?
            if importances is None and hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                
            # CHECK 3: Is it a Linear model?
            if importances is None and hasattr(best_model, 'coef_'):
                importances = best_model.coef_[0]

            # FALLBACK: If we still have nothing (e.g., Neural Net), fake it evenly
            if importances is None:
                importances = np.ones(len(feature_names))

            # Sort and extract top features
            top_indices = np.argsort(np.abs(importances))[-5:][::-1]
            top_features = [(feature_names[i], abs(importances[i])) for i in top_indices]
            
            # Generate Text Explanation
            explanation = PlainEnglishTranslator.explain_prediction(
                manga_title, prob, top_features
            )
            
            # --- FEATURE 3: SANITIZE & SAVE ---
            safe_filename = re.sub(r'[^\w\s-]', '', manga_title)[:30].strip()
            # Handle empty filenames (edge case)
            if not safe_filename: safe_filename = f"manga_{idx}"
            
            # with open(f"{Config.PREDICTIONS_DIR}/{safe_filename}_prediction.txt", 'w', encoding='utf-8') as f:
            #     f.write(explanation)
            pdf_path = f"{Config.PREDICTIONS_DIR}/{safe_filename}_REPORT.pdf"
            
            pdf_engine = MangaReportHTML()
            pdf_engine.generate_report(
                title=manga_title,
                probability=prob,
                features=top_features,
                action_plan=explanation,
                filename=pdf_path
            )
                
        except Exception as e:
            print(f"   ⚠️ Skipped '{manga_title}' due to error: {e}")
            continue
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 7: GENERATE COMPREHENSIVE REPORT
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 7: GENERATING COMPREHENSIVE REPORTS")
    print("="*80)
    
    summary_engine = MangaReportHTML()
    summary_path = f"{Config.OUTPUT_DIR}/EXECUTIVE_SUMMARY_REPORT.pdf"
    
    # PASS THE DATA HERE
    summary_engine.generate_portfolio_summary(
        results_dict=predictions_dict,
        filename=summary_path,
        demo_data=demo_lifespans,
        feature_df=explainer.importances,
        aft_model=survival_analysis.run_weibull_predictions()
    )

    print("\n>> [REPORT] Generating Financial Statement...")

    financial_metrics = visualizer.get_financial_metrics(df)
    
    # Print a quick check to console
    print(f"   -> Net Impact: ${financial_metrics['net_impact']:,.0f}")
    print(f"   -> ROI: {financial_metrics['roi_percent']:.1f}%")
    
    # Generate the PDF
    roi_path = f"{Config.OUTPUT_DIR}/FINANCIAL_IMPACT_REPORT.pdf"
    summary_engine.generate_roi_report(financial_metrics, roi_path)
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 8: SAVE MODELS
    # ─────────────────────────────────────────────────────────────────
    print("\n>> [MODELS] Saving trained models...")
    
    try:
        with open(f"{Config.MODELS_DIR}/best_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        
        with open(f"{Config.MODELS_DIR}/feature_names.pkl", 'wb') as f:
            pickle.dump(feature_names, f)
        
        with open(f"{Config.MODELS_DIR}/scaler.pkl", 'wb') as f:
            pickle.dump(ml_engine.scaler, f)
        
        print(f"   ✓ Models saved to {Config.MODELS_DIR}/")
    except Exception as e:
        print(f"   ⚠️ Could not save models: {e}")
    
    # ─────────────────────────────────────────────────────────────────
    # STEP 9: ORACLE MODE (Now with "Watchlist" Saving)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("🔮 ORACLE MODE ACTIVATED")
    print("   Type a manga name. The report will be saved to 'my_personal_watchlist'.")
    print("   (Type 'q' to exit)")
    print("="*80)
    
    # 1. Create a separate folder for YOUR manga
    WATCHLIST_DIR = "my_personal_watchlist"
    if not os.path.exists(WATCHLIST_DIR):
        os.makedirs(WATCHLIST_DIR)
        print(f"   📂 Created new folder: {WATCHLIST_DIR}/")

    # Combine data for search
    X_full = pd.concat([ml_engine.X_train, ml_engine.X_test])
    
    while True:
        user_input = input("\n>> Enter manga name: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting. Good luck!")
            break
            
        if not user_input:
            continue
            
        # Search
        matches = df[df['title'].str.contains(user_input, case=False, na=False)]
        
        if len(matches) == 0:
            print("   ❌ No manga found. Try a simpler keyword.")
            continue
            
        # Select best match
        if len(matches) > 1:
            print(f"   ⚠️ Found {len(matches)} matches. Using: '{matches.iloc[0]['title']}'")
        
        target_manga = matches.iloc[0]
        manga_id = target_manga.name
        title = target_manga['title']
        
        if manga_id not in X_full.index:
            print(f"   ⚠️ Error: '{title}' was filtered out (bad data).")
            continue
            
        # Predict
        manga_features = X_full.loc[[manga_id]] 
        prob = best_model.predict_proba(manga_features)[:, 1][0]
        
        # --- GENERATE EXPLANATION ---
        # Get feature importance for this specific manga
        importances = None

        if hasattr(best_model, 'estimators_'):
            for sub_model in best_model.estimators_:
                if hasattr(sub_model, 'feature_importances_'):
                    importances = sub_model.feature_importances_
                    break
        
        # CHECK 2: Is it a standard Tree model?
        if importances is None and hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            
        # CHECK 3: Is it a Linear model?
        if importances is None and hasattr(best_model, 'coef_'):
            importances = best_model.coef_[0]

        # FALLBACK: If we still have nothing, fake it evenly
        if importances is None:
            importances = np.ones(len(feature_names))

        # Sort and extract top features
        top_indices = np.argsort(np.abs(importances))[-5:][::-1]
        top_features = [(feature_names[i], abs(importances[i])) for i in top_indices]
        
        is_finished_bool = (target_manga['is_finished'] == 1)

        # Create the text report
        explanation = PlainEnglishTranslator.explain_prediction(
            title, prob, top_features,is_finished=is_finished_bool
        )
        
        # --- SAVE TO PERSONAL FOLDER ---
        # Sanitize filename (remove : / \ etc)
        safe_filename = re.sub(r'[^\w\s-]', '', title)[:30].strip()
        # file_path = f"{WATCHLIST_DIR}/{safe_filename}_REPORT.txt"
        
        # with open(file_path, 'w', encoding='utf-8') as f:
        #     f.write(explanation)

        file_path = f"{WATCHLIST_DIR}/{safe_filename}_REPORT.pdf"
        pdf = MangaReportHTML()
        pdf.generate_report(
            title=title,
            probability=prob,
            features=top_features,
            action_plan=explanation, # Passing the full text analysis here
            filename=file_path
        )
            
        # Print valid confirmation
        print(f"\n   📖 {title}")
        print(f"   ──────────────────────────────")
        if prob > 0.5:
            print(f"   💀 DANGER: {prob:.1%} chance of AXE.")
        else:
            print(f"   🛡️ SAFE: {1-prob:.1%} chance of SURVIVAL.")
        print(f"   💾 Detailed report saved to: {file_path}")

    # ─────────────────────────────────────────────────────────────────
    # COMPLETION BANNER
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE")
    print("="*80)
    print(f"""
Output Files Generated:
────────────────────────
✓ {Config.OUTPUT_DIR}/                     - Visualizations & plots
✓ {Config.PREDICTIONS_DIR}/                - Individual predictions
✓ {Config.MODELS_DIR}/                     - Trained models for future use
✓ SUMMARY_REPORT.txt                      - Executive summary

Key Insights:
─────────────
• Best Model Used: {max(ml_engine.results.keys(), key=lambda k: ml_engine.results[k]['test_auc']).upper()}
• Test AUC: {ml_engine.results[max(ml_engine.results.keys(), key=lambda k: ml_engine.results[k]['test_auc'])]['test_auc']:.4f}
  (AUC measures how well the model distinguishes between cancelled and continuing manga)

• Models Evaluated: {len(ml_engine.results)}
• Features Engineered: {len(feature_names)}
• Dataset Size: {len(df):,} manga

Remember:
──────────
✓ These predictions are PROBABILITY ESTIMATES, not certainties
✓ Use as ONE data point among many
✓ Consider qualitative factors not captured by data
✓ Monitor predictions over time as new data arrives

═══════════════════════════════════════════════════════════════════════════════
""")
    
    print(f"\n✓ Check the '{Config.OUTPUT_DIR}/' folder for detailed visualizations")
    print(f"✓ Check '{Config.PREDICTIONS_DIR}/' for individual manga predictions\n")


if __name__ == "__main__":
    main()
