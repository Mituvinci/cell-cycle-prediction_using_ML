"""
Data Loading and Preprocessing Utilities
=========================================

Contains functions for:
- Data loading from CSV files
- Feature selection (ElasticNet, SelectKBest)
- Scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Class balancing (SMOTE, ADASYN, undersampling)

Author: Halima Akhter
Date: 2025-11-24
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import ElasticNetCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids


def apply_scaling(X, method='standard', scaler=None):
    """
    Apply feature scaling to the data.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    method : str, default='standard'
        Scaling method: 'standard', 'minmax', or 'robust'
    scaler : sklearn scaler, optional
        Pre-fitted scaler to use. If None, creates and fits a new one.

    Returns:
    --------
    X_scaled : np.ndarray
        Scaled features
    scaler : sklearn scaler
        Fitted scaler object
    """
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def elasticnet_feature_selection(X_scaled, y, alphas=np.logspace(-3, 1, 10), l1_ratio=0.5):
    """
    Performs feature selection using ElasticNetCV.

    This EXACTLY matches the original implementation from 1_0_principle_aurelien_ml.py

    Args:
        X_scaled (pd.DataFrame): Scaled features.
        y (pd.Series): Target labels (encoded).
        alphas (array): Range of alpha values (log-space recommended).
        l1_ratio (float): L1 ratio for ElasticNet (0=Ridge, 1=Lasso).

    Returns:
        pd.Index: Selected feature names.
    """
    # Fit ElasticNet with cross-validation
    enet = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=5,
        max_iter=10000,  # Higher for stability
        tol=1e-5,  # Precision in convergence
        n_jobs=-1  # Utilize all CPUs
    )
    enet.fit(X_scaled, y)

    # Select features with non-zero coefficients (NO threshold, just non-zero)
    selected_features = np.abs(enet.coef_) > 0

    return X_scaled.columns[selected_features]  # Return selected feature names


def preprocess_rna_data(data, scaling_method, selection_method=None):
    """
    Preprocesses RNA data: filtering, feature selection, ADASYN, undersampling, and scaling.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 479-562

    Args:
        data (pd.DataFrame): Raw RNA-seq data.
        scaling_method (str): Scaling method to use ('standard', 'minmax', 'robust').
        selection_method (str, optional): Feature selection method ('ElasticCV', 'SelectKBest', or None).

    Returns:
        tuple: (X_train_resampled, X_test_scaled, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    label_encoder = LabelEncoder()

    # Step 1: Filter out rows with missing 'Predicted' labels
    data_with_label = data.dropna(subset=['Predicted'])

    # Extract labels and cell IDs
    y_labeled = data_with_label['Predicted'].reset_index(drop=True)
    cell_ids_labeled = data_with_label['gex_barcode'].reset_index(drop=True)  # Keep cell IDs for test set

    # Remove non-numeric columns except 'Predicted'
    non_numeric_cols = data_with_label.select_dtypes(include=['object', 'category']).columns
    non_numeric_cols = [col for col in non_numeric_cols if col != 'Predicted']
    data_with_label = data_with_label.drop(columns=non_numeric_cols)

    # Drop the 'Predicted' column from features
    X_labeled = data_with_label.drop(columns=['Predicted']).reset_index(drop=True)

    # Step 2: Train-Test Split (80-20)
    X_train, X_test, y_train, y_test, _, cell_ids_test = train_test_split(
        X_labeled, y_labeled, cell_ids_labeled, test_size=0.2, random_state=42, stratify=y_labeled, shuffle=True
    )


    # Step 4: Feature Selection (Only Apply Scaling If Needed)
    ElasticCV_m = "ElasticCV"
    SelectKBest_m = "SelectKBest"

    if selection_method in [ElasticCV_m, SelectKBest_m]:
        X_train_scaled, _ = apply_scaling(X_train, method=scaling_method)  # Temporary scaling for feature selection
        print(type(X_train_scaled))

        # Convert to DataFrame (Ensure Column Names Exist)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
        if selection_method == ElasticCV_m:
            selected_features = elasticnet_feature_selection(X_train_scaled, y_train_encoded)

        elif selection_method == SelectKBest_m:
            selector = SelectKBest(f_classif, k=300)
            selector.fit(X_train_scaled, y_train_encoded)
            selected_features = X_train_scaled.columns[selector.get_support()]

    else:
        selected_features = X_train.columns  # Use all features if no selection

    # Step 5: Apply Selected Features
    X_train = X_train[selected_features].copy()
    X_test = X_test[selected_features].copy()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Step 6: Apply SMOTE (COMMENTED OUT in original - keeping for reference)
    """
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)

    # Step 7: Apply Undersampling
    min_count = min(Counter(y_train_resampled).values())
    undersample = RandomUnderSampler(sampling_strategy={cls: min_count for cls in Counter(y_train_resampled).keys()}, random_state=42)
    X_train_resampled, y_train_resampled = undersample.fit_resample(X_train_resampled, y_train_resampled)
    """

    # Step 7: Apply Scaling **BEFORE** ADASYN
    X_train_scaled, scaler = apply_scaling(X_train, method=scaling_method)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Step 8: Apply ADASYN (Adaptive Synthetic Sampling)
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train_encoded)

    # Step 9: Apply Cluster-Based Undersampling
    cluster_undersample = ClusterCentroids(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = cluster_undersample.fit_resample(X_train_resampled, y_train_resampled)

    # Step 10: Return Processed Data
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder


def load_and_preprocess_data(scaling_method, is_reh=True, check_feature=False, selection_method=None):
    """
    Loads and preprocesses the REH and SUP datasets (combined 80% + 20%).

    EXACT implementation from 1_0_principle_aurelien_ml.py line 568-639

    Args:
        scaling_method (str): Scaling method to use.
        is_reh (bool): True for REH, False for SUP.
        check_feature (bool): Whether to check feature overlap (not implemented here).
        selection_method (str): Feature selection method.

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    # Paths to your processed RNA data files
    path_reh = (
        "D:/Halima's Data/Thesis_2/RCode/filtered_result/re_assign_cc/reh/"
        "filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"
    )
    path_sup = (
        "D:/Halima's Data/Thesis_2/RCode/filtered_result/re_assign_cc/sup/"
        "filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv"
    )

    # Load the RNA datasets
    data_reh = pd.read_csv(path_reh)
    data_sup = pd.read_csv(path_sup)

    # Find common features between REH and SUP
    training_features_set = set(data_reh.columns) & set(data_sup.columns)

    # NOTE: Benchmark data loading and feature matching commented out
    # as it's part of the original but not needed for basic preprocessing
    """
    path_gse146773_benchmark = (
        "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/"
        "GSE146773_seurat_normalized_gene_expression.csv"
    )
    data_gse146773_benchmark = pd.read_csv(path_gse146773_benchmark)
    data_gse146773_benchmark.rename(columns={'paper_phase': 'Predicted', 'cell': 'gex_barcode'}, inplace=True)

    path_gse64016_benchmark = (
        "D:/Halima's Data/Thesis_2/RCode/Cell_Cycle_prediction_with_scATAC_Seq/paper1/Training_data/Benchmark_data/"
        "GSE64016_seurat_normalized_gene_expression.csv"
    )
    data_gse64016_benchmark = pd.read_csv(path_gse64016_benchmark)
    data_gse64016_benchmark.rename(columns={'Labeled': 'Predicted'}, inplace=True)

    # Find common features before preprocessing
    common_features = list((set(data_reh.columns) & set(data_sup.columns)) &
                           (set(X_labeled_gse146773.columns) & set(X_labeled_gse64016.columns)))
    common_features.extend(['Predicted', 'gex_barcode'])
    data_reh = data_reh[common_features]
    data_sup = data_sup[common_features]
    """

    # Preprocess REH and SUP with the same feature set
    if is_reh:
        X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder = preprocess_rna_data(
            data_reh, scaling_method, selection_method
        )
    else:
        X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder = preprocess_rna_data(
            data_sup, scaling_method, selection_method
        )

    return X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder
