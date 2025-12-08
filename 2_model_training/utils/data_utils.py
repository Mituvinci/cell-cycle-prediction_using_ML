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

#######################################################
#           BENCHMARK DATA CONFIGURATION              #
#######################################################
# Set to True to use integrated benchmark data (scTQuery REH-aligned)
# Set to False to use original benchmark data
USE_INTEGRATED_BENCHMARKS = True

reh_path = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/REH_aligned_formatted"
pbmc_path = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/PBMC_aligned_formatted"

mouse_brain_path = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/scTQuery/MouseBrain_formatted"

# Integrated benchmark paths (from scTQuery REH alignment)
INTEGRATED_BENCHMARK_DIR = reh_path
# Original benchmark paths
ORIGINAL_BENCHMARK_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data"
ORIGINAL_BUETTNER_DIR = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"


def match_scaler_feature_format(df, scaler, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'phase', 'Phase']):
    """
    Automatically detects the scaler's feature name format and applies the same format to DataFrame.

    Supports:
    - ALL UPPERCASE: "AAAS", "ACTB" (old human models)
    - Title Case: "Aaas", "Actb" (new 7-dataset models)
    - all lowercase: "aaas", "actb" (rare, but supported)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with gene columns
    scaler : sklearn scaler object
        Fitted scaler with feature_names_in_ attribute
    exclude_cols : list
        Columns to exclude from transformation (metadata columns)

    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names matching scaler format
    """
    # Get a sample feature from scaler to detect format
    if not hasattr(scaler, 'feature_names_in_') or len(scaler.feature_names_in_) == 0:
        return df

    sample_feature = str(scaler.feature_names_in_[0])

    # Detect format
    if sample_feature.isupper():
        # ALL UPPERCASE format (old models)
        target_format = 'upper'
    elif sample_feature[0].isupper() and not sample_feature.isupper():
        # Title Case format (new models)
        target_format = 'capitalize'
    elif sample_feature.islower():
        # all lowercase format
        target_format = 'lower'
    else:
        # Mixed case, assume capitalize as default
        target_format = 'capitalize'

    # Apply the detected format
    rename_dict = {}
    for col in df.columns:
        if col not in exclude_cols:
            if target_format == 'upper':
                transformed = col.upper()
            elif target_format == 'capitalize':
                transformed = col.capitalize()
            elif target_format == 'lower':
                transformed = col.lower()
            else:
                transformed = col

            if transformed != col:
                rename_dict[col] = transformed

    # Rename columns
    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


def uppercase_gene_names(df, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'phase', 'Phase', 'Unnamed: 0', 'dataset', 'Labeled', 'paper_phase']):
    """
    Converts all gene names in a DataFrame to UPPERCASE.

    This ensures consistent gene naming across all datasets:
    - Mouse genes: GNAI3, PBSN, CDC45
    - Human genes: GAPDH, ACTB, TP53

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with gene columns
    exclude_cols : list
        Columns to exclude from transformation (metadata columns)

    Returns:
    --------
    pd.DataFrame
        DataFrame with UPPERCASE gene names
    """
    # Create a mapping of old names to UPPERCASE names
    rename_dict = {}
    for col in df.columns:
        if col not in exclude_cols:
            upper = col.upper()
            if upper != col:
                rename_dict[col] = upper

    # Rename columns
    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


def capitalize_gene_names(df, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'phase', 'Phase']):
    """
    DEPRECATED: Use uppercase_gene_names() instead.

    This function is kept for backward compatibility but now calls uppercase_gene_names().
    All gene names should be ALL UPPERCASE for consistency.
    """
    return uppercase_gene_names(df, exclude_cols)


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

    # Step 7: Save column names before scaling (apply_scaling returns numpy array)
    feature_columns = X_train.columns.tolist()

    # Step 8: Apply Scaling **BEFORE** ADASYN
    X_train_scaled, scaler = apply_scaling(X_train, method=scaling_method)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Step 9: Apply ADASYN (Adaptive Synthetic Sampling)
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train_encoded)

    # Step 10: Apply Cluster-Based Undersampling
    cluster_undersample = ClusterCentroids(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = cluster_undersample.fit_resample(X_train_resampled, y_train_resampled)

    # Step 11: Convert back to DataFrame (ADASYN/ClusterCentroids return numpy arrays)
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_columns)

    # Step 11: Return Processed Data
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder


def load_custom_training_data(custom_data_path, scaling_method, selection_method=None):
    """
    Loads and preprocesses custom training data from user-provided CSV file.

    Expected CSV format:
        - First column: cell_id (or gex_barcode)
        - Second column: phase_label (must be 'G1', 'S', or 'G2M')
        - Remaining columns: gene expression values

    Args:
        custom_data_path (str): Path to custom training data CSV
        scaling_method (str): Scaling method to use
        selection_method (str): Feature selection method

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    print(f"\nLoading custom training data from: {custom_data_path}")

    # Load custom data
    data_custom = pd.read_csv(custom_data_path)

    # Check if data already has standard column names (from merged training data)
    if 'gex_barcode' in data_custom.columns and 'Predicted' in data_custom.columns:
        # Data is already in correct format (merged training data)
        print(f"  Detected merged training data format")
    else:
        # Data needs column renaming (simple custom format: cell_id, phase_label, genes...)
        first_col = data_custom.columns[0]
        second_col = data_custom.columns[1]

        # Rename to standard format
        data_custom.rename(columns={first_col: 'gex_barcode', second_col: 'Predicted'}, inplace=True)
        print(f"  Renamed columns: {first_col} → gex_barcode, {second_col} → Predicted")

    # Capitalize all gene names for species independence
    data_custom = capitalize_gene_names(data_custom)

    print(f"  Loaded {len(data_custom)} cells")
    print(f"  Features: {len(data_custom.columns) - 2} genes (excluding gex_barcode, Predicted)")

    # Get phase distribution (ensure it's a Series)
    phase_series = data_custom['Predicted']
    if isinstance(phase_series, pd.DataFrame):
        # If somehow 'Predicted' returns a DataFrame, take the first column
        phase_series = phase_series.iloc[:, 0]
    print(f"  Phase distribution: {phase_series.value_counts().to_dict()}")

    # Validate phase labels
    valid_phases = {'G1', 'S', 'G2M'}
    unique_phases = set(phase_series.unique())
    if not unique_phases.issubset(valid_phases):
        raise ValueError(f"Invalid phase labels found: {unique_phases}. Must be one of {valid_phases}")

    # Preprocess using existing function
    X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder = preprocess_rna_data(
        data_custom, scaling_method, selection_method
    )

    return X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder


def load_and_preprocess_data(scaling_method, is_reh=True, check_feature=False, selection_method=None, custom_data_path=None):
    """
    Loads and preprocesses the REH and SUP datasets (combined 80% + 20%) OR custom data.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 568-639
    Extended to support custom training data.

    Args:
        scaling_method (str): Scaling method to use.
        is_reh (bool): True for REH, False for SUP (ignored if custom_data_path provided).
        check_feature (bool): Whether to check feature overlap (not implemented here).
        selection_method (str): Feature selection method.
        custom_data_path (str, optional): Path to custom training CSV. If provided, uses custom data instead of REH/SUP.

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    # If custom data path provided, use it
    if custom_data_path is not None:
        return load_custom_training_data(custom_data_path, scaling_method, selection_method)

    # Otherwise, use default REH/SUP data
    # Paths to your processed RNA data files (dynamically find project root)
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, "data")

    path_reh = os.path.join(
        data_dir,
        "filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"
    )
    path_sup = os.path.join(
        data_dir,
        "filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv"
    )

    # Load the RNA datasets
    data_reh = pd.read_csv(path_reh)
    data_sup = pd.read_csv(path_sup)

    # Capitalize all gene names for species independence
    data_reh = capitalize_gene_names(data_reh)
    data_sup = capitalize_gene_names(data_sup)

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


def load_gene_list(gene_list_path):
    """
    Loads gene names from a text file.

    Args:
        gene_list_path (str): Path to text file with one gene per line (UPPERCASE)

    Returns:
        list: Sorted list of gene names (UPPERCASE)
    """
    with open(gene_list_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return sorted(genes)


def load_and_preprocess_data(scaling_method, dataset='hpsc', gene_list_path=None, selection_method=None):
    """
    Loads and preprocesses data using a pre-computed gene list.

    This function:
    1. Loads training data based on dataset parameter
    2. Converts ALL gene names to UPPERCASE
    3. Filters to genes from provided gene_list_path
    4. Applies preprocessing (scaling, ADASYN, undersampling)

    Args:
        scaling_method (str): Scaling method to use.
        dataset (str): Which training data to use ('hpsc', 'pbmc', 'mouse_brain', 'reh', 'sup').
        gene_list_path (str): Path to text file with gene names (one per line, UPPERCASE)
        selection_method (str): Feature selection method.

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    import os

    if gene_list_path is None:
        raise ValueError("gene_list_path is required. Provide path to gene list file.")

    # Load gene list
    print("=" * 60)
    print("LOADING GENE LIST FROM FILE")
    print("=" * 60)
    gene_list = load_gene_list(gene_list_path)
    print(f"Gene list path: {gene_list_path}")
    print(f"Number of genes: {len(gene_list)}")
    print(f"First 5 genes: {gene_list[:5]}")
    print(f"Last 5 genes: {gene_list[-5:]}")

    # Paths to training datasets
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"

    path_reh = f"{data_dir}/training_data_1_GD428_21136_Hu_REH_Parental_normalized_gene_expression.csv"
    path_sup = f"{data_dir}/training_data_2_GD444_21136_Hu_Sup_Parental_normalized_gene_expression.csv"
    path_pbmc = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv"
    path_mouse_brain = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv"
    path_hpsc = f"{data_dir}/GSE75748_hPSC_final_training_matrix.csv"

    # Dataset path mapping
    dataset_paths = {
        'hpsc': path_hpsc,
        'pbmc': path_pbmc,
        'mouse_brain': path_mouse_brain,
        'reh': path_reh,
        'sup': path_sup
    }

    if dataset not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'hpsc', 'pbmc', 'mouse_brain', 'reh', or 'sup'")

    # Load selected training dataset
    print(f"\nLoading training data: {dataset}")
    selected_data = pd.read_csv(dataset_paths[dataset])
    print(f"  Loaded: {len(selected_data)} cells, {len(selected_data.columns)} columns")

    # Convert ALL gene names to UPPERCASE
    print("\nConverting all gene names to UPPERCASE...")
    selected_data = uppercase_gene_names(selected_data)

    # Get available feature columns
    metadata_exclude = ['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'phase', 'Phase',
                        'Unnamed: 0', 'dataset', 'Labeled', 'paper_phase']

    available_features = [col for col in selected_data.columns if col not in metadata_exclude]
    available_features_set = set(available_features)

    # Check gene overlap
    gene_list_set = set(gene_list)
    missing_genes = gene_list_set - available_features_set
    if missing_genes:
        print(f"  WARNING: {len(missing_genes)} genes from gene list not found in dataset")
        print(f"  Missing genes (first 10): {list(missing_genes)[:10]}")
        # Filter gene list to only available genes
        gene_list = [g for g in gene_list if g in available_features_set]
        print(f"  Using {len(gene_list)} genes that are available")

    # Filter to gene list + metadata
    metadata_cols = ['Predicted', 'gex_barcode']
    existing_metadata = [col for col in metadata_cols if col in selected_data.columns]
    selected_data = selected_data[gene_list + existing_metadata]

    print(f"\nFiltered training data shape: {selected_data.shape}")
    print(f"Number of genes used for training: {len(gene_list)}")

    # Apply preprocessing to selected training data
    X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder = preprocess_rna_data(
        selected_data, scaling_method, selection_method
    )

    return X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder


#######################################################
#           BENCHMARK DATA LOADING FUNCTIONS          #
#######################################################

def scaling_benchmark_simple(benchmark_data, scaler, debug=False):
    """
    Applies scaling to benchmark data using ONLY the fitted scaler (CORRECT METHOD).

    This matches how training data is scaled:
    1. Training: scaler.fit_transform(X_train) - scales using training statistics
    2. Benchmark: scaler.transform(X_benchmark) - scales using SAME training statistics

    NO additional normalization - keeps benchmark in same distribution as training data.

    Args:
        benchmark_data (pd.DataFrame): Benchmark data to transform.
        scaler: Fitted scaler object.
        debug (bool): Print scaling statistics for verification.

    Returns:
        pd.DataFrame: Scaled benchmark data (using training statistics)
    """
    # Ensure benchmark data has the same feature order as scaler expects
    expected_feature_order = scaler.feature_names_in_
    benchmark_data = benchmark_data[expected_feature_order]

    if debug:
        print(f"\n[SCALING DEBUG - SIMPLE METHOD]")
        print(f"Before scaling: mean={benchmark_data.mean().mean():.4f}, std={benchmark_data.std().mean():.4f}")

    # Apply scaling ONLY (no additional normalization)
    scaled_benchmark_data = scaler.transform(benchmark_data)

    # Convert scaled data back to DataFrame
    scaled_benchmark_data = pd.DataFrame(scaled_benchmark_data, columns=benchmark_data.columns)

    if debug:
        print(f"After scaling:  mean={scaled_benchmark_data.mean().mean():.4f}, std={scaled_benchmark_data.std().mean():.4f}")
        print(f"Note: Mean/std NOT exactly 0/1 because using TRAINING scaler (correct!)")

    return scaled_benchmark_data


def scaling_benchmark(benchmark_data, scaler):
    """
    OLD METHOD - Applies scaling to benchmark data using a fitted scaler, followed by normalization.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 327-352

    CRITICAL: This function performs TWO steps:
    1. Apply scaler.transform()
    2. Apply additional normalization: (scaled - mean) / std

    WARNING: This creates distribution mismatch between training and benchmark data.
    Use scaling_benchmark_simple() instead for correct evaluation.

    Args:
        benchmark_data (pd.DataFrame): Benchmark data to transform.
        scaler: Fitted scaler object.

    Returns:
        pd.DataFrame: Scaled and normalized benchmark data
    """
    # Ensure benchmark data has the same feature order as scaler expects
    expected_feature_order = scaler.feature_names_in_
    benchmark_data = benchmark_data[expected_feature_order]

    # Apply scaling
    scaled_benchmark_data = scaler.transform(benchmark_data)

    # Convert scaled data back to DataFrame
    scaled_benchmark_data = pd.DataFrame(scaled_benchmark_data, columns=benchmark_data.columns)

    # CRITICAL STEP: Apply additional normalization
    X_benchmark_normalized = (scaled_benchmark_data - scaled_benchmark_data.mean()) / scaled_benchmark_data.std()

    # Convert back to DataFrame
    X_benchmark_normalized = pd.DataFrame(X_benchmark_normalized, columns=benchmark_data.columns)

    return X_benchmark_normalized


def preprocess_benchmark_data(data, title, check_feature=False):
    """
    Preprocesses a benchmark data file (basic preprocessing).

    EXACT implementation from 1_0_principle_aurelien_ml.py line 644-676

    Args:
        data (pd.DataFrame): Raw benchmark data.
        title (str): Dataset name for debugging.
        check_feature (bool): Whether to check feature overlap (not implemented here).

    Returns:
        tuple: (X_labeled, y_labeled, cell_ids_labeled)
    """
    # Drop rows with NA in 'Predicted'
    data_with_label = data.dropna(subset=['Predicted'])

    # Identify non-numeric columns
    non_numeric_cols = data_with_label.select_dtypes(include=['object', 'category']).columns

    # Prepare the feature matrix (X) and labels (y)
    y_labeled = data_with_label['Predicted'].reset_index(drop=True)
    cell_ids_labeled = data_with_label['gex_barcode'].reset_index(drop=True)
    X_labeled = data_with_label.drop(columns=non_numeric_cols).reset_index(drop=True)

    # Replace special JSON characters in feature names
    X_labeled.columns = [
        col.replace('{', '').replace('}', '').replace(':', '') for col in X_labeled.columns
    ]

    return (X_labeled, y_labeled, cell_ids_labeled)


def data_preprocess_GSE(data, scaler, dataset_name, check_feature=False, is_old_model=False, scaling_method='simple'):
    """
    Preprocesses a GSE benchmark data file.

    Args:
        data (pd.DataFrame): Raw GSE data.
        scaler: Fitted scaler object.
        dataset_name (str): Dataset name.
        check_feature (bool): Whether to check feature overlap.
        is_old_model (bool): If True, skip capitalization (for old models). Default False.
        scaling_method (str): 'simple' (no double normalization) or 'double' (old method). Default 'simple'.

    Returns:
        tuple: (X_labeled, y_labeled, cell_ids_labeled)
               Processed features, labels, and cell IDs.
    """
    X_labeled, y_labeled, cell_ids_labeled = preprocess_benchmark_data(data, dataset_name, check_feature)

    # For old models: NO capitalization, keep original gene names
    # For new models: Match feature format to scaler
    if not is_old_model:
        X_labeled = match_scaler_feature_format(X_labeled, scaler)

    # Get expected features
    expected_features = scaler.feature_names_in_

    # For old models: Handle missing genes with mean imputation
    # For new models: Raise KeyError if genes missing
    if is_old_model:
        missing_genes = set(expected_features) - set(X_labeled.columns)
        if missing_genes:
            print(f"  {dataset_name}: {len(missing_genes)} missing genes, using zero imputation")
            # Fill missing genes with 0 (safe for TML models like AdaBoost)
            missing_df = pd.DataFrame(0, index=X_labeled.index, columns=list(missing_genes))
            X_labeled = pd.concat([X_labeled, missing_df], axis=1)

        # Reorder to match scaler
        X_labeled = X_labeled[expected_features]
    else:
        # New models: strict matching, raise error if missing
        X_labeled = X_labeled[expected_features]

    # Apply scaling based on chosen method
    if scaling_method == 'simple':
        X_labeled = scaling_benchmark_simple(X_labeled, scaler)
    elif scaling_method == 'double':
        X_labeled = scaling_benchmark(X_labeled, scaler)
    else:
        raise ValueError(f"Invalid scaling_method: {scaling_method}. Use 'simple' or 'double'.")

    # Ensure no NaN values (both old and new models)
    if X_labeled.isna().any().any():
        nan_count = X_labeled.isna().sum().sum()
        print(f"  WARNING: {dataset_name} has {nan_count} NaN values after scaling, filling with 0")
        X_labeled = X_labeled.fillna(0)

    return X_labeled, y_labeled, cell_ids_labeled


def load_reh_or_sup_benchmark(scaler, reh_sup="sup", is_old_model=False, scaling_method='simple'):
    """
    Loads REH or SUP benchmark data.

    Args:
        scaler: Fitted scaler object.
        reh_sup (str): "reh" or "sup" to select dataset.
        is_old_model (bool): If True, skip capitalization (for old models). Default False.
        scaling_method (str): 'simple' (no double normalization) or 'double' (old method). Default 'simple'.

    Returns:
        tuple: (X_labeled, y_labeled, cell_ids_labeled)
    """
    import os

    # Dynamically find project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, "data")

    if reh_sup == "reh":
        path = os.path.join(
            data_dir,
            "filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"
        )
    else:
        path = os.path.join(
            data_dir,
            "filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv"
        )

    # Load the RNA datasets
    data = pd.read_csv(path)

    data_with_label = data.dropna(subset=['Predicted'])

    # Extract labels and cell IDs
    y_labeled = data_with_label['Predicted'].reset_index(drop=True)
    cell_ids_labeled = data_with_label['gex_barcode'].reset_index(drop=True)

    # Remove non-numeric columns except 'Predicted'
    non_numeric_cols = data_with_label.select_dtypes(include=['object', 'category']).columns
    non_numeric_cols = [col for col in non_numeric_cols if col != 'Predicted']
    data_with_label = data_with_label.drop(columns=non_numeric_cols)

    # Drop the 'Predicted' column from features
    X_labeled = data_with_label.drop(columns=['Predicted']).reset_index(drop=True)

    # Clean up column names
    X_labeled.columns = [
        col.replace('{', '').replace('}', '').replace(':', '') for col in X_labeled.columns
    ]

    # For old models: NO capitalization, keep original gene names
    # For new models: Match feature format to scaler
    if not is_old_model:
        print(f"  [DEBUG] {reh_sup.upper()} - BEFORE match_scaler_feature_format:")
        print(f"    First 5 genes in benchmark: {list(X_labeled.columns[:5])}")
        print(f"    Scaler expects format sample: {scaler.feature_names_in_[:5]}")

        X_labeled = match_scaler_feature_format(X_labeled, scaler)

        print(f"  [DEBUG] {reh_sup.upper()} - AFTER match_scaler_feature_format:")
        print(f"    First 5 genes in benchmark: {list(X_labeled.columns[:5])}")

    # Ensure that the return value is a DataFrame
    X_labeled = pd.DataFrame(X_labeled, columns=X_labeled.columns)

    # Get expected features
    expected_features = scaler.feature_names_in_

    print(f"  [DEBUG] {reh_sup.upper()} - Feature matching:")
    print(f"    Genes in benchmark: {len(X_labeled.columns)}")
    print(f"    Genes expected by model: {len(expected_features)}")
    print(f"    Missing from benchmark: {len(set(expected_features) - set(X_labeled.columns))}")

    # For old models: Handle missing genes with mean imputation
    # For new models: Raise KeyError if genes missing
    if is_old_model:
        missing_genes = set(expected_features) - set(X_labeled.columns)
        if missing_genes:
            print(f"  {reh_sup.upper()}: {len(missing_genes)} missing genes, using zero imputation")
            # Fill missing genes with 0 (safe for TML models like AdaBoost)
            missing_df = pd.DataFrame(0, index=X_labeled.index, columns=list(missing_genes))
            X_labeled = pd.concat([X_labeled, missing_df], axis=1)

        # Reorder to match scaler
        X_labeled = X_labeled[expected_features]
    else:
        # New models: strict matching, raise error if missing
        X_labeled = X_labeled[expected_features]

    # Apply scaling based on chosen method
    if scaling_method == 'simple':
        X_labeled = scaling_benchmark_simple(X_labeled, scaler)
    elif scaling_method == 'double':
        X_labeled = scaling_benchmark(X_labeled, scaler)
    else:
        raise ValueError(f"Invalid scaling_method: {scaling_method}. Use 'simple' or 'double'.")

    # Ensure no NaN values (both old and new models)
    if X_labeled.isna().any().any():
        nan_count = X_labeled.isna().sum().sum()
        print(f"  WARNING: {reh_sup.upper()} has {nan_count} NaN values after scaling, filling with 0")
        X_labeled = X_labeled.fillna(0)

    return X_labeled, y_labeled, cell_ids_labeled


def load_gse146773(scaler, check_feature=False, is_old_model=False, scaling_method='simple'):
    """
    Loads and preprocesses the GSE146773 benchmark data.

    IMPORTANT:
    - All gene names are converted to UPPERCASE
    - Gene columns are sorted ALPHABETICALLY to match scaler

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.
        is_old_model (bool): If True, skip capitalization (for old models). Default False.
        scaling_method (str): 'simple' (no double normalization) or 'double' (old method). Default 'simple'.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Select benchmark data directory based on configuration
    if USE_INTEGRATED_BENCHMARKS:
        data_dir = INTEGRATED_BENCHMARK_DIR
        print(f"  [INFO] Using INTEGRATED benchmark: GSE146773")
    else:
        data_dir = ORIGINAL_BENCHMARK_DIR
        print(f"  [INFO] Using ORIGINAL benchmark: GSE146773")

    path_gse_benchmark = os.path.join(
        data_dir,
        "GSE146773_seurat_normalized_gene_expression.csv"
    )
    data_gse_benchmark = pd.read_csv(path_gse_benchmark)

    # Rename columns to keep consistent naming
    data_gse_benchmark.rename(columns={'paper_phase': 'Predicted', 'cell': 'gex_barcode'}, inplace=True)

    # Convert ALL gene names to UPPERCASE
    data_gse_benchmark = uppercase_gene_names(data_gse_benchmark)

    data_gse_benchmark['Predicted'] = data_gse_benchmark['Predicted'].str.replace(
        r'^S.*', 'S', regex=True
    )
    data_gse_benchmark = data_gse_benchmark.dropna(subset=['Predicted'])

    # Preprocess the benchmark data
    benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
        data_gse_benchmark, scaler, "GSE146773", check_feature, is_old_model, scaling_method
    )

    return benchmark_features, benchmark_labels, benchmark_cell_ids


def load_gse64016(scaler, check_feature=False, is_old_model=False, scaling_method='simple'):
    """
    Loads and preprocesses the GSE64016 benchmark data.

    IMPORTANT:
    - All gene names are converted to UPPERCASE
    - Gene columns are sorted ALPHABETICALLY to match scaler

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.
        is_old_model (bool): If True, skip capitalization (for old models). Default False.
        scaling_method (str): 'simple' (no double normalization) or 'double' (old method). Default 'simple'.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Select benchmark data directory based on configuration
    if USE_INTEGRATED_BENCHMARKS:
        data_dir = INTEGRATED_BENCHMARK_DIR
        print(f"  [INFO] Using INTEGRATED benchmark: GSE64016")
    else:
        data_dir = ORIGINAL_BENCHMARK_DIR
        print(f"  [INFO] Using ORIGINAL benchmark: GSE64016")

    path_gse_benchmark = os.path.join(
        data_dir,
        "GSE64016_seurat_normalized_gene_expression.csv"
    )
    data_gse_benchmark = pd.read_csv(path_gse_benchmark)

    # Rename columns to keep consistent naming
    data_gse_benchmark.rename(columns={'Labeled': 'Predicted'}, inplace=True)

    # Convert ALL gene names to UPPERCASE
    data_gse_benchmark = uppercase_gene_names(data_gse_benchmark)

    # Remove rows where 'Predicted' starts with 'H1'
    data_gse_benchmark = data_gse_benchmark[
        ~data_gse_benchmark['Predicted'].str.startswith('H1')
    ]

    # Replace 'Predicted' values based on their prefixes
    data_gse_benchmark['Predicted'] = data_gse_benchmark['Predicted'].str.replace(
        r'^G2.*', 'G2M', regex=True
    )
    data_gse_benchmark['Predicted'] = data_gse_benchmark['Predicted'].str.replace(
        r'^G1.*', 'G1', regex=True
    )
    data_gse_benchmark['Predicted'] = data_gse_benchmark['Predicted'].str.replace(
        r'^S.*', 'S', regex=True
    )

    # Preprocess the benchmark data
    benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
        data_gse_benchmark, scaler, "GSE64016", check_feature, is_old_model, scaling_method
    )

    return benchmark_features, benchmark_labels, benchmark_cell_ids


def load_buettner_mesc(scaler, check_feature=False, is_old_model=False, scaling_method='simple'):
    """
    Loads and preprocesses the Buettner mESC benchmark data.

    IMPORTANT:
    - All gene names are converted to UPPERCASE
    - Gene columns are sorted ALPHABETICALLY to match scaler

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.
        is_old_model (bool): If True, capitalize + mean imputation for missing genes. Default False.
        scaling_method (str): 'simple' (no double normalization) or 'double' (old method). Default 'simple'.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Select benchmark data directory based on configuration
    if USE_INTEGRATED_BENCHMARKS:
        data_dir = INTEGRATED_BENCHMARK_DIR
        print(f"  [INFO] Using INTEGRATED benchmark: Buettner_mESC")
    else:
        data_dir = ORIGINAL_BUETTNER_DIR
        print(f"  [INFO] Using ORIGINAL benchmark: Buettner_mESC")

    # Use cleaned benchmark data (without embedded Phase column)
    path_buettner_benchmark = os.path.join(
        data_dir,
        "Buettner_mESC_benchmark_clean.csv"
    )

    # Load ground truth labels (always from original location)
    path_buettner_ground_truth = os.path.join(
        ORIGINAL_BUETTNER_DIR,
        "Buettner_mESC_goundTruth.csv"
    )

    # Load expression data and ground truth
    data_buettner_benchmark = pd.read_csv(path_buettner_benchmark)
    ground_truth = pd.read_csv(path_buettner_ground_truth)

    # Convert ALL gene names to UPPERCASE
    data_buettner_benchmark = uppercase_gene_names(data_buettner_benchmark)

    # Merge with ground truth to get labels
    data_buettner_benchmark = data_buettner_benchmark.merge(
        ground_truth[['Cell_ID', 'Phase']],
        on='Cell_ID',
        how='left'
    )

    # Rename columns to keep consistent naming
    data_buettner_benchmark.rename(columns={'Phase': 'Predicted', 'Cell_ID': 'gex_barcode'}, inplace=True)

    # Standardize phase labels (if needed)
    # Map common variations to standard G1, S, G2M format
    phase_mapping = {
        'G1': 'G1',
        'S': 'S',
        'G2': 'G2M',
        'G2M': 'G2M',
        'M': 'G2M'
    }

    data_buettner_benchmark['Predicted'] = data_buettner_benchmark['Predicted'].map(
        lambda x: phase_mapping.get(x, x)
    )
    data_buettner_benchmark = data_buettner_benchmark.dropna(subset=['Predicted'])

    # OLD MODELS: Capitalize + mean imputation for missing genes
    # NEW MODELS: Match scaler format
    if is_old_model:
        # Always capitalize for Buettner with old models
        data_buettner_benchmark = match_scaler_feature_format(data_buettner_benchmark, scaler)

        # Extract features and labels
        X_labeled, y_labeled, cell_ids_labeled = preprocess_benchmark_data(
            data_buettner_benchmark, "Buettner_mESC", check_feature
        )

        # Handle missing genes with zero imputation
        expected_features = scaler.feature_names_in_
        missing_genes = set(expected_features) - set(X_labeled.columns)

        if missing_genes:
            print(f"  Buettner missing {len(missing_genes)} genes, using zero imputation")
            # Fill missing genes with 0 (safe for TML models like AdaBoost)
            missing_df = pd.DataFrame(0, index=X_labeled.index, columns=list(missing_genes))
            X_labeled = pd.concat([X_labeled, missing_df], axis=1)

        # Reorder to match scaler
        X_labeled = X_labeled[expected_features]

        # Apply scaling based on chosen method
        if scaling_method == 'simple':
            benchmark_features = scaling_benchmark_simple(X_labeled, scaler)
        elif scaling_method == 'double':
            benchmark_features = scaling_benchmark(X_labeled, scaler)
        else:
            raise ValueError(f"Invalid scaling_method: {scaling_method}. Use 'simple' or 'double'.")
        benchmark_labels = y_labeled
        benchmark_cell_ids = cell_ids_labeled
    else:
        # New models: use standard preprocessing
        benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
            data_buettner_benchmark, scaler, "Buettner_mESC", check_feature, is_old_model, scaling_method
        )

    # Ensure no NaN values (both old and new models)
    if benchmark_features.isna().any().any():
        nan_count = benchmark_features.isna().sum().sum()
        print(f"  WARNING: Buettner has {nan_count} NaN values, filling with 0")
        benchmark_features = benchmark_features.fillna(0)

    return benchmark_features, benchmark_labels, benchmark_cell_ids


def load_custom_benchmark(custom_benchmark_path, scaler, benchmark_name="CustomBenchmark", check_feature=False):
    """
    Loads and preprocesses custom benchmark data from user-provided CSV file.

    Expected CSV format (same as training data):
        - First column: cell_id (or gex_barcode)
        - Second column: phase_label (must be 'G1', 'S', or 'G2M')
        - Remaining columns: gene expression values

    Args:
        custom_benchmark_path (str): Path to custom benchmark CSV
        scaler: Fitted scaler object
        benchmark_name (str): Name for this benchmark (used in logging)
        check_feature (bool): Whether to check feature overlap

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    print(f"\nLoading custom benchmark data from: {custom_benchmark_path}")

    # Load custom benchmark
    data_custom_benchmark = pd.read_csv(custom_benchmark_path)

    # Detect cell_id and phase_label columns
    first_col = data_custom_benchmark.columns[0]
    second_col = data_custom_benchmark.columns[1]

    # Rename to standard format
    data_custom_benchmark.rename(columns={first_col: 'gex_barcode', second_col: 'Predicted'}, inplace=True)

    # Match feature format to scaler (auto-detects UPPERCASE vs Title Case)
    data_custom_benchmark = match_scaler_feature_format(data_custom_benchmark, scaler)

    print(f"  Loaded {len(data_custom_benchmark)} cells")
    print(f"  Features: {len(data_custom_benchmark.columns) - 2} genes")
    print(f"  Phase distribution: {data_custom_benchmark['Predicted'].value_counts().to_dict()}")

    # Validate phase labels
    valid_phases = {'G1', 'S', 'G2M'}
    unique_phases = set(data_custom_benchmark['Predicted'].unique())
    if not unique_phases.issubset(valid_phases):
        # Try to map common variations
        phase_mapping = {
            'G1': 'G1',
            'S': 'S',
            'G2': 'G2M',
            'G2M': 'G2M',
            'M': 'G2M'
        }
        data_custom_benchmark['Predicted'] = data_custom_benchmark['Predicted'].map(
            lambda x: phase_mapping.get(x, x)
        )
        data_custom_benchmark = data_custom_benchmark.dropna(subset=['Predicted'])

        # Validate again after mapping
        unique_phases = set(data_custom_benchmark['Predicted'].unique())
        if not unique_phases.issubset(valid_phases):
            raise ValueError(f"Invalid phase labels found: {unique_phases}. Must be one of {valid_phases}")

    # Preprocess the benchmark data
    benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
        data_custom_benchmark, scaler, benchmark_name, check_feature
    )

    return benchmark_features, benchmark_labels, benchmark_cell_ids
