import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

def get_task_type(y):
    """Determine if the task is classification or regression."""
    if y.dtype == 'object' or y.nunique() < 20: # heuristic
        return 'classification'
    else:
        return 'regression'

def preprocess_data(df, target_col, feature_cols=None):
    """Advanced preprocessing pipeline with feature engineering."""
    df = df.copy()
    
    # Drop rows where the target column is missing, since we cannot train without labels
    df = df.dropna(subset=[target_col])
    
    # Separate features and target
    if feature_cols is not None:
        # Ensure target_col is NOT in features to avoid data leakage
        clean_features = [f for f in feature_cols if f != target_col]
        if len(clean_features) == 0:
            raise ValueError("No feature columns selected. Please select at least one feature to train the model.")
        X = df[clean_features]
    else:
        X = df.drop(columns=[target_col])
        
    y = df[target_col]
    
    # Handle target encoding for classification if target is string
    task_type = get_task_type(y)
    label_encoder = None
    if task_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    
    # Impute numerical missing values with mean
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
    # Impute categorical missing values with mode
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Remove near-zero variance features (noise columns)
    try:
        vt = VarianceThreshold(threshold=0.01)
        X_filtered = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])
        if X_filtered.shape[1] > 0:
            X = X_filtered
    except Exception:
        pass  # Keep all features if variance filter fails
    
    # Drop highly correlated features (>0.95 correlation)
    try:
        if len(X.columns) > 1:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
            if len(drop_cols) < len(X.columns):  # Don't drop all columns
                X = X.drop(columns=drop_cols)
    except Exception:
        pass
        
    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    preprocessors = {
        'num_imputer': num_imputer if len(num_cols) > 0 else None,
        'cat_imputer': cat_imputer if len(cat_cols) > 0 else None,
        'scaler': scaler,
        'final_columns': list(X_scaled.columns),
        'num_cols': list(num_cols),
        'cat_cols': list(cat_cols)
    }
    
    return X_scaled, y, task_type, label_encoder, preprocessors

def auto_tune_model(name, base_model, task_type):
    """Returns a RandomizedSearchCV wrapper with expanded hyperparameter grids."""
    param_grid = {}
    n_iter = 10  # More iterations for better tuning
    
    if 'Random Forest' in name:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        n_iter = 15
    elif 'XGBoost' in name:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        n_iter = 20
    elif 'Gradient Boosting' in name:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        n_iter = 15
    elif 'Extra Trees' in name:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        n_iter = 12
    elif 'AdaBoost' in name:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
        n_iter = 10
    elif 'KNN' in name:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        n_iter = 10
    elif 'SVC' in name or 'SVR' in name:
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        n_iter = 10
    elif 'MLP' in name:
        param_grid = {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        n_iter = 10
    elif 'Decision Tree' in name:
        param_grid = {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'] if task_type == 'classification' else ['squared_error', 'friedman_mse']
        }
        n_iter = 12
    elif 'Ridge' in name or 'Lasso' in name:
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        }
        n_iter = 6
    else:
        return base_model  # Don't tune others
        
    scoring_metric = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == 'classification' else 5
    
    return RandomizedSearchCV(
        base_model, param_distributions=param_grid, n_iter=n_iter, cv=cv_strategy,
        scoring=scoring_metric, n_jobs=-1, random_state=42
    )

def train_and_evaluate(X, y, task_type, apply_improvements=False):
    """Train multiple models and return results, with optional advanced auto-tuning."""
    
    # Use stratified split for classification to preserve class ratios
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    if apply_improvements and task_type == 'classification' and SMOTE is not None:
        try:
            # Using auto strategy but maintaining random_state for consistency
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=min(5, min(pd.Series(y_train).value_counts()) - 1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print("SMOTE ignored due to constraints (e.g., too few samples per class):", e)
            
    results = []
    best_model = None
    best_score = -1 if task_type == 'classification' else float('inf')
    best_model_name = ""
    
    if task_type == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'MLP Neural Net': MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
            'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
            'SVC': SVC(random_state=42)
        }
    else:
        models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_jobs=-1),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'MLP Neural Net': MLPRegressor(max_iter=500, random_state=42, early_stopping=True),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Linear Regression': LinearRegression(n_jobs=-1),
            'SVR': SVR()
        }
        
    trained_models = {}
    tuning_info = {}  # Track parameters used by each model
    last_error = None
    
    for name, model in models.items():
        try:
            if apply_improvements:
                model = auto_tune_model(name, model, task_type)
                
            model.fit(X_train, y_train)
            
            # If it was wrapped in a tuner, extract the best estimator and params
            if hasattr(model, 'best_estimator_'):
                final_predictor = model.best_estimator_
                tuning_info[name] = {
                    'tuned': True,
                    'best_params': model.best_params_,
                    'best_cv_score': round(model.best_score_, 4),
                    'n_iter': model.n_splits_,
                }
            else:
                final_predictor = model
                # Capture key params for baseline models
                params = final_predictor.get_params()
                # Filter out less useful params for display
                filtered = {k: v for k, v in params.items() if k not in ['verbose', 'warm_start', 'n_jobs', 'random_state', 'class_weight']}
                tuning_info[name] = {
                    'tuned': False,
                    'default_params': filtered
                }
            
            y_pred = final_predictor.predict(X_test)
            
            trained_models[name] = final_predictor
            
            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred)
                avg_type = 'weighted' if len(np.unique(y)) > 2 else 'binary'
                prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
                rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
                f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
                
                results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1 Score': f1
                })
                
                if acc > best_score: # prioritize Accuracy when evaluating best model
                    best_score = acc
                    best_model = final_predictor
                    best_model_name = name
                    
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MSE': mse,
                    'MAE': mae,
                    'R2 Score': r2
                })
                
                if rmse < best_score: # lower error is better
                    best_score = rmse
                    best_model = final_predictor
                    best_model_name = name
        except Exception as e:
            last_error = e
            print(f"Error training {name}: {e}")
            
    if len(results) == 0:
        raise ValueError(f"All models failed to train. Last model error: {last_error}")
        
    results_df = pd.DataFrame(results)
    
    # Sort by best metric so best model is at top
    if task_type == 'classification':
        results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    else:
        results_df = results_df.sort_values('RMSE', ascending=True).reset_index(drop=True)
    
    return results_df, best_model_name, best_model, trained_models, X_test, y_test, tuning_info
