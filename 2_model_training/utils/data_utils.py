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


def capitalize_gene_names(df, exclude_cols=['gex_barcode', 'Predicted', 'Cell_ID', 'cell', 'phase', 'Phase']):
    """
    Capitalizes all gene names in a DataFrame (first letter uppercase, rest lowercase).

    This ensures species-independent gene naming:
    - Mouse genes: Gnai3, Pbsn, Cdc45
    - Human genes: Gapdh, Actb, Tp53

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with gene columns
    exclude_cols : list
        Columns to exclude from capitalization (metadata columns)

    Returns:
    --------
    pd.DataFrame
        DataFrame with capitalized gene names
    """
    # Create a mapping of old names to capitalized names
    rename_dict = {}
    for col in df.columns:
        if col not in exclude_cols:
            # Capitalize: first letter uppercase, rest lowercase
            capitalized = col.capitalize()
            if capitalized != col:
                rename_dict[col] = capitalized

    # Rename columns
    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


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


def load_and_preprocess_data_v2(scaling_method, dataset='new_human', check_feature=False, selection_method=None):
    """
    Loads and preprocesses data with 7-dataset feature intersection.

    CRITICAL: Computes intersection of ALL 7 datasets BEFORE preprocessing:
    - REH, SUP-B15 (old human training)
    - GSE146773, GSE64016 (human benchmarks)
    - Buettner_mESC (mouse benchmark)
    - new_human_PBMC (new human training)
    - new_mouse_brain (new mouse training)

    Then selects appropriate training data based on dataset parameter.

    Args:
        scaling_method (str): Scaling method to use.
        dataset (str): Which training data to use ('new_human', 'new_mouse', 'reh', 'sup').
        check_feature (bool): Whether to check feature overlap.
        selection_method (str): Feature selection method.

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder)
    """
    # Paths to all datasets
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"

    path_reh = f"{data_dir}/filtered_normalized_gene_expression_cc_label1_GD428_21136_Hu_REH_Parental_overlapped_all_four_regions.csv"
    path_sup = f"{data_dir}/filtered_normalized_gene_expression_cc_label2_GD444_21136_Hu_Sup_Parental_overlapped_all_four_regions.csv"
    path_gse146773 = f"{data_dir}/GSE146773_seurat_normalized_gene_expression.csv"
    path_gse64016 = f"{data_dir}/GSE64016_seurat_normalized_gene_expression.csv"
    path_buettner = f"{data_dir}/Buettner_mESC_benchmark_clean.csv"
    path_new_human = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_human/pbmc_human_training_data.csv"
    path_new_mouse = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/cell_cycle_prediction/1_consensus_labeling/assign/final_training_data_mouse/mouse_brain_training_data.csv"

    # Load all 7 datasets
    print("Loading all 7 datasets...")
    data_reh = pd.read_csv(path_reh)
    data_sup = pd.read_csv(path_sup)
    data_gse146773 = pd.read_csv(path_gse146773)
    data_gse64016 = pd.read_csv(path_gse64016)
    data_buettner = pd.read_csv(path_buettner)
    data_new_human = pd.read_csv(path_new_human)
    data_new_mouse = pd.read_csv(path_new_mouse)

    # Capitalize ALL gene names for consistent matching
    print("Capitalizing gene names for consistent matching...")
    data_reh = capitalize_gene_names(data_reh)
    data_sup = capitalize_gene_names(data_sup)
    data_gse146773 = capitalize_gene_names(data_gse146773)
    data_gse64016 = capitalize_gene_names(data_gse64016)
    data_buettner = capitalize_gene_names(data_buettner)
    data_new_human = capitalize_gene_names(data_new_human)
    data_new_mouse = capitalize_gene_names(data_new_mouse)

    # Rename columns to standardize
    data_gse146773.rename(columns={'paper_phase': 'Predicted', 'cell': 'gex_barcode'}, inplace=True)
    data_gse64016.rename(columns={'Labeled': 'Predicted'}, inplace=True)

    # Get feature columns (exclude metadata)
    def get_feature_cols(df):
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
        feature_cols = [col for col in df.columns if col not in non_numeric_cols]
        return set(feature_cols)

    # Compute intersection of ALL 7 datasets
    print("Computing feature intersection across all 7 datasets...")
    common_features = (
        get_feature_cols(data_reh) &
        get_feature_cols(data_sup) &
        get_feature_cols(data_gse146773) &
        get_feature_cols(data_gse64016) &
        get_feature_cols(data_buettner) &
        get_feature_cols(data_new_human) &
        get_feature_cols(data_new_mouse)
    )

    common_features = list(common_features)
    print(f"Found {len(common_features)} common features across all 7 datasets")

    if len(common_features) == 0:
        raise ValueError("ERROR: No common features found across all 7 datasets! Check gene name formatting.")

    # Add metadata columns back
    metadata_cols = ['Predicted', 'gex_barcode']

    # Filter all datasets to common features
    def filter_to_common(df, common_feats, metadata):
        existing_metadata = [col for col in metadata if col in df.columns]
        return df[common_feats + existing_metadata]

    data_reh = filter_to_common(data_reh, common_features, metadata_cols)
    data_sup = filter_to_common(data_sup, common_features, metadata_cols)
    data_gse146773 = filter_to_common(data_gse146773, common_features, metadata_cols)
    data_gse64016 = filter_to_common(data_gse64016, common_features, metadata_cols)
    data_buettner = filter_to_common(data_buettner, common_features, metadata_cols)
    data_new_human = filter_to_common(data_new_human, common_features, metadata_cols)
    data_new_mouse = filter_to_common(data_new_mouse, common_features, metadata_cols)

    # Select training dataset based on parameter
    dataset_map = {
        'new_human': data_new_human,
        'new_mouse': data_new_mouse,
        'reh': data_reh,
        'sup': data_sup
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'new_human', 'new_mouse', 'reh', or 'sup'")

    selected_data = dataset_map[dataset]
    print(f"Using training data: {dataset}")

    # Apply preprocessing to selected training data
    X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder = preprocess_rna_data(
        selected_data, scaling_method, selection_method
    )

    return X_train_resampled, X_test, y_train_resampled, y_test, cell_ids_test, scaler, label_encoder


#######################################################
#           BENCHMARK DATA LOADING FUNCTIONS          #
#######################################################

def scaling_benchmark(benchmark_data, scaler):
    """
    Applies scaling to benchmark data using a fitted scaler, followed by normalization.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 327-352

    CRITICAL: This function performs TWO steps:
    1. Apply scaler.transform()
    2. Apply additional normalization: (scaled - mean) / std

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


def data_preprocess_GSE(data, scaler, dataset_name, check_feature=False):
    """
    Preprocesses a GSE benchmark data file.

    EXACT implementation from user's original code.
    Simply filters to expected features - NO adding zeros for missing genes!

    Args:
        data (pd.DataFrame): Raw GSE data.
        scaler: Fitted scaler object.
        dataset_name (str): Dataset name.
        check_feature (bool): Whether to check feature overlap.

    Returns:
        tuple: (X_labeled, y_labeled, cell_ids_labeled)
               Processed features, labels, and cell IDs.
    """
    X_labeled, y_labeled, cell_ids_labeled = preprocess_benchmark_data(data, dataset_name, check_feature)

    # Simply filter to expected features (matching scaler)
    # If features are missing, this will raise KeyError - which is correct!
    expected_features = scaler.feature_names_in_
    X_labeled = X_labeled[expected_features]

    # Apply scaling
    X_labeled = scaling_benchmark(X_labeled, scaler)

    return X_labeled, y_labeled, cell_ids_labeled


def load_reh_or_sup_benchmark(scaler, reh_sup="sup"):
    """
    Loads REH or SUP benchmark data.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 702-742

    Args:
        scaler: Fitted scaler object.
        reh_sup (str): "reh" or "sup" to select dataset.

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

    # Capitalize all gene names for species independence
    data = capitalize_gene_names(data)

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

    # Ensure that the return value is a DataFrame
    X_labeled = pd.DataFrame(X_labeled, columns=data_with_label.drop(columns=['Predicted']).columns)
    X_labeled = X_labeled[scaler.feature_names_in_]
    X_labeled = scaling_benchmark(X_labeled, scaler)

    return X_labeled, y_labeled, cell_ids_labeled


def load_gse146773(scaler, check_feature=False):
    """
    Loads and preprocesses the GSE146773 benchmark data.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 745-786

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Use absolute path to benchmark data
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data"

    path_gse_benchmark = os.path.join(
        data_dir,
        "GSE146773_seurat_normalized_gene_expression.csv"
    )
    data_gse_benchmark = pd.read_csv(path_gse_benchmark)

    # Rename columns to keep consistent naming
    data_gse_benchmark.rename(columns={'paper_phase': 'Predicted', 'cell': 'gex_barcode'}, inplace=True)

    # NOTE: NOT capitalizing gene names for old model compatibility
    # Old models were trained with original gene name format (uppercase for human)
    # data_gse_benchmark = capitalize_gene_names(data_gse_benchmark)

    data_gse_benchmark['Predicted'] = data_gse_benchmark['Predicted'].str.replace(
        r'^S.*', 'S', regex=True
    )
    data_gse_benchmark = data_gse_benchmark.dropna(subset=['Predicted'])

    # Preprocess the benchmark data
    benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
        data_gse_benchmark, scaler, "GSE146773", check_feature
    )

    return benchmark_features, benchmark_labels, benchmark_cell_ids


def load_gse64016(scaler, check_feature=False):
    """
    Loads and preprocesses the GSE64016 benchmark data.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 789-842

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Use absolute path to benchmark data
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data"

    path_gse_benchmark = os.path.join(
        data_dir,
        "GSE64016_seurat_normalized_gene_expression.csv"
    )
    data_gse_benchmark = pd.read_csv(path_gse_benchmark)

    # Rename columns to keep consistent naming
    data_gse_benchmark.rename(columns={'Labeled': 'Predicted'}, inplace=True)

    # NOTE: NOT capitalizing gene names for old model compatibility
    # Old models were trained with original gene name format (uppercase for human)
    # data_gse_benchmark = capitalize_gene_names(data_gse_benchmark)

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
        data_gse_benchmark, scaler, "GSE64016", check_feature
    )

    return benchmark_features, benchmark_labels, benchmark_cell_ids


def load_buettner_mesc(scaler, check_feature=False):
    """
    Loads and preprocesses the Buettner mESC benchmark data.

    Args:
        scaler: Fitted scaler object.
        check_feature (bool): Whether to check feature overlap.

    Returns:
        tuple: (benchmark_features, benchmark_labels, benchmark_cell_ids)
    """
    import os
    from collections import Counter

    # Use absolute path to benchmark data
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data"

    # Use cleaned benchmark data (without embedded Phase column)
    path_buettner_benchmark = os.path.join(
        data_dir,
        "Buettner_mESC_benchmark_clean.csv"
    )

    # Load ground truth labels
    path_buettner_ground_truth = os.path.join(
        data_dir,
        "Buettner_mESC_goundTruth.csv"
    )

    # Load expression data and ground truth
    data_buettner_benchmark = pd.read_csv(path_buettner_benchmark)
    ground_truth = pd.read_csv(path_buettner_ground_truth)

    # Merge with ground truth to get labels
    data_buettner_benchmark = data_buettner_benchmark.merge(
        ground_truth[['Cell_ID', 'Phase']],
        on='Cell_ID',
        how='left'
    )

    # Rename columns to keep consistent naming
    data_buettner_benchmark.rename(columns={'Phase': 'Predicted', 'Cell_ID': 'gex_barcode'}, inplace=True)

    # NOTE: NOT capitalizing gene names for old model compatibility
    # Old models were trained with original gene name format
    # data_buettner_benchmark = capitalize_gene_names(data_buettner_benchmark)

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

    # Preprocess the benchmark data
    benchmark_features, benchmark_labels, benchmark_cell_ids = data_preprocess_GSE(
        data_buettner_benchmark, scaler, "Buettner_mESC", check_feature
    )

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

    # Capitalize all gene names for species independence
    data_custom_benchmark = capitalize_gene_names(data_custom_benchmark)

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
