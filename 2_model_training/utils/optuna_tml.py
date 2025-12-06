"""
Optuna Optimization for Traditional ML Models
==============================================

Contains Optuna hyperparameter optimization for:
- AdaBoost
- Random Forest
- LightGBM
- Ensemble (combination of all three)

EXACT implementation from 2_0_principle_aurelien_ml_traditional.py

Author: Halima Akhter
Date: 2025-11-24
"""

import optuna
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


def objective(trial, X_train_inner, y_train_inner, X_val, y_val, model_type):
    """
    Optuna objective function for optimizing AdaBoost, Random Forest, and LGBM.

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 311-353

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
    X_train_inner : pd.DataFrame
        Training features from inner CV split
    y_train_inner : np.ndarray
        Training labels from inner CV split
    X_val : pd.DataFrame
        Validation features from inner CV split
    y_val : np.ndarray
        Validation labels from inner CV split
    model_type : str
        Model type: 'adaboost', 'random_forest', or 'lgbm'

    Returns:
    --------
    float
        Error (1 - accuracy) to minimize
    """

    if model_type == "adaboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        }
        model = AdaBoostClassifier(**params, algorithm="SAMME")

    elif model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
        model = RandomForestClassifier(**params, class_weight="balanced")

    elif model_type == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        }
        model = LGBMClassifier(**params, device="cpu")

    else:
        raise ValueError("Invalid model type")

    # Train model
    model.fit(X_train_inner, y_train_inner)

    # Evaluate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return 1 - accuracy  # Minimize error (maximize accuracy)


def optimize_traditional_model(X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, model_type, n_trials=20):
    """
    Runs Optuna optimization for a given traditional model.

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 356-378

    Parameters:
    -----------
    X_train : pd.DataFrame
        Full training features (outer fold)
    y_train : np.ndarray
        Full training labels (outer fold)
    X_train_inner : pd.DataFrame
        Inner training features (for optimization)
    y_train_inner : np.ndarray
        Inner training labels (for optimization)
    X_val : pd.DataFrame
        Validation features (for optimization)
    y_val : np.ndarray
        Validation labels (for optimization)
    model_type : str
        Model type: 'adaboost', 'random_forest', or 'lgbm'
    n_trials : int, default=20
        Number of Optuna trials

    Returns:
    --------
    tuple
        (best_model, best_params)
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_inner, y_train_inner, X_val, y_val, model_type), n_trials=n_trials)

    best_params = study.best_params
    best_trial = study.best_trial

    print(f"Best Params: {best_params}")

    if model_type == "adaboost":
        best_model = AdaBoostClassifier(**best_params, algorithm="SAMME")
    elif model_type == "random_forest":
        best_model = RandomForestClassifier(**best_params, class_weight="balanced")
    elif model_type == "lgbm":
        best_model = LGBMClassifier(**best_params, device="cpu")

    best_model.fit(X_train, y_train)

    return best_model, best_params


def optimize_ensemble(X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, n_trials=20):
    """
    Optimizes AdaBoost, RF, LGBM separately and creates an ensemble.

    EXACT implementation from 2_0_principle_aurelien_ml_traditional.py line 380-404

    Parameters:
    -----------
    X_train : pd.DataFrame
        Full training features (outer fold)
    y_train : np.ndarray
        Full training labels (outer fold)
    X_train_inner : pd.DataFrame
        Inner training features (for optimization)
    y_train_inner : np.ndarray
        Inner training labels (for optimization)
    X_val : pd.DataFrame
        Validation features (for optimization)
    y_val : np.ndarray
        Validation labels (for optimization)
    n_trials : int, default=20
        Number of Optuna trials per model

    Returns:
    --------
    tuple
        (ensemble_model, params_dict)
        ensemble_model: VotingClassifier with soft voting
        params_dict: {"AdaBoost": ada_params, "RF": rf_params, "LGBM": lgbm_params}
    """
    print("\n[Optimizing AdaBoost]")
    ada_model, ada_params = optimize_traditional_model(X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, "adaboost", n_trials)

    print("\n[Optimizing Random Forest]")
    rf_model, rf_params = optimize_traditional_model(X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, "random_forest", n_trials)

    print("\n[Optimizing LGBM]")
    lgbm_model, lgbm_params = optimize_traditional_model(X_train, y_train, X_train_inner, y_train_inner, X_val, y_val, "lgbm", n_trials)

    ensemble_model = VotingClassifier(
        estimators=[
            ("ada", ada_model),
            ("rf", rf_model),
            ("lgbm", lgbm_model)
        ],
        voting="soft"
    )

    ensemble_model.fit(X_train, y_train)

    return ensemble_model, {"AdaBoost": ada_params, "RF": rf_params, "LGBM": lgbm_params}
