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

    # Detect cell_id and phase_label columns
    first_col = data_custom.columns[0]
    second_col = data_custom.columns[1]

    # Rename to standard format
    data_custom.rename(columns={first_col: 'gex_barcode', second_col: 'Predicted'}, inplace=True)

    print(f"  Loaded {len(data_custom)} cells")
    print(f"  Features: {len(data_custom.columns) - 2} genes")
    print(f"  Phase distribution: {data_custom['Predicted'].value_counts().to_dict()}")

    # Validate phase labels
    valid_phases = {'G1', 'S', 'G2M'}
    unique_phases = set(data_custom['Predicted'].unique())
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


#######################################################
#           BENCHMARK DATA LOADING FUNCTIONS          #
#######################################################

def scaling_benchmark(benchmark_data, scaler):
    """
    Applies scaling to benchmark data using a fitted scaler.

    EXACT implementation from 1_0_principle_aurelien_ml.py line 327-352

    Args:
        benchmark_data (pd.DataFrame): Benchmark data to transform.
        scaler: Fitted scaler object.

    Returns:
        pd.DataFrame: Scaled benchmark data
    """
    # Ensure benchmark data has the same feature order as scaler expects
    expected_feature_order = scaler.feature_names_in_
    benchmark_data = benchmark_data[expected_feature_order]

    # Apply scaling
    scaled_benchmark_data = scaler.transform(benchmark_data)

    # Convert scaled data back to DataFrame
    scaled_benchmark_data = pd.DataFrame(scaled_benchmark_data, columns=benchmark_data.columns)

    return scaled_benchmark_data


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

    Handles missing features by adding them as zeros.
    Based on data_preprocess_GSE_local from 1_0_principle_aurelien_ml.py line 871-896

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

    expected_features = scaler.feature_names_in_
    current_features = X_labeled.columns

    # Drop extra genes that are in benchmark but not in training
    extra_features = [f for f in current_features if f not in expected_features]
    if extra_features:
        X_labeled = X_labeled.drop(columns=extra_features)

    # Identify missing genes that are in training but not in benchmark
    missing_features = [f for f in expected_features if f not in current_features]
    if missing_features:
        # Add missing features with zeros (using concat for performance)
        missing_df = pd.DataFrame(0, index=X_labeled.index, columns=missing_features)
        X_labeled = pd.concat([X_labeled, missing_df], axis=1)

    # Reorder columns to match expected feature order
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
    data_dir = "/users/ha00014/Halimas_projects/DeepLearning_CellCyelPhaseDetection_scRNASeq/data/Training_data/Benchmark_data"

    path_buettner_benchmark = os.path.join(
        data_dir,
        "Buettner_mESC_SeuratNormalized_ML_ready.csv"
    )
    data_buettner_benchmark = pd.read_csv(path_buettner_benchmark)

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
