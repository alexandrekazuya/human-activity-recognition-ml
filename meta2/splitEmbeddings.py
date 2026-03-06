import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skrebate import ReliefF
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ficheiros.outliers import normalizeZscore, normalizeZscore_with_stats

def load_split_arrays(embeddings_path="meta2/embeddings.npy", y_path="meta2/y.npy", participants_path="meta2/participants.npy"):
    X = np.load(embeddings_path)
    y = np.load(y_path) #y é oarray de labels (atividades)
    participants = np.load(participants_path)
    return X, y, participants

def split60_20_20(X, y, participants, random_state=42):
    # first split train/test (80/20)
    X_train, X_test, y_train, y_test, part_train, part_test = train_test_split(
        X, y, participants, test_size=0.2, random_state=random_state, stratify=y)
    # split train into train/val (60/20)
    X_train, X_val, y_train, y_val, part_train, part_val = train_test_split(
        X_train, y_train, part_train,
        test_size=0.25,   # 0.25 de 0.8 = 0.2 total
        random_state=random_state,
        stratify=y_train
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "part_train": part_train,
        "X_val": X_val,
        "y_val": y_val,
        "part_val": part_val,
        "X_test": X_test,
        "y_test": y_test,
        "part_test": part_test
    }

def split933(X, y, participants):
    # 9-3-3 split between subjects
    unique_subjects = np.unique(participants)

    train_subjects = unique_subjects[:9]
    val_subjects = unique_subjects[9:12]
    test_subjects = unique_subjects[12:15]

    train_mask = np.isin(participants, train_subjects)
    val_mask = np.isin(participants, val_subjects)
    test_mask = np.isin(participants, test_subjects)

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "part_train": participants[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "part_val": participants[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "part_test": participants[test_mask]
    }

def create_scenario_all_features(X_train, X_val, X_test, y_train, y_val, y_test):
    return {
        "X_train": X_train.copy(),
        "X_val": X_val.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_val": y_val.copy(),
        "y_test": y_test.copy()
    }

def create_scenario_pca_90(X_train, X_val, X_test, y_train, y_val, y_test, variance_threshold=0.9):
    
    X_train_norm, mean, safe = normalizeZscore_with_stats(X_train)
    X_val_norm = (X_val - mean) / safe
    X_test_norm = (X_test - mean) / safe

    pca_full = PCA()
    pca_full.fit(X_train_norm)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    if np.any(cum_var >= variance_threshold):
        n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    else:
        n_components = pca_full.n_components_

    pca_90 = PCA(n_components=n_components)
    pca_90.fit(X_train_norm)

    scenario = {
        "X_train": pca_90.transform(X_train_norm),
        "X_val": pca_90.transform(X_val_norm),
        "X_test": pca_90.transform(X_test_norm),
        "n_components": n_components,
        "explained_variance_ratio": pca_90.explained_variance_ratio_,
        "pca": pca_90,
        "train_mean": mean,
        "train_std": safe,
        "y_train": y_train.copy(),
        "y_val": y_val.copy(),
        "y_test": y_test.copy()
    }
    return scenario

def create_scenario_relieff_top15(X_train, X_val, X_test, y_train, y_val, y_test, do_zscore=True):
   
    X_train_norm, mean, safe = normalizeZscore_with_stats(X_train)
    X_val_norm = (X_val - mean) / safe
    X_test_norm = (X_test - mean) / safe

    relief = ReliefF(n_neighbors=5, n_features_to_select=15, n_jobs=-1)
    relief.fit(X_train_norm, y_train)
    top15_idx = relief.top_features_[:15]
    scenario = {
        "X_train": X_train_norm[:, top15_idx],
        "X_val": X_val_norm[:, top15_idx],
        "X_test": X_test_norm[:, top15_idx],
        "top15_idx": top15_idx,
        "train_mean": mean,
        "train_std": safe,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    return scenario

if __name__ == "__main__":
    X, y, participants = load_split_arrays()
    split_602020 = split60_20_20(X, y, participants)
    split_933 = split933(X, y, participants)

    # Para o split 60/20/20
    scenario_a_602020 = create_scenario_all_features(
        split_602020["X_train"],
        split_602020["X_val"],
        split_602020["X_test"],
        split_602020["y_train"],
        split_602020["y_val"],
        split_602020["y_test"]
    ) 
    np.save("meta2/scenarios/embeddings_scenario_a_602020.npy", scenario_a_602020)

    # Para o split 9/3/3
    scenario_a_933 = create_scenario_all_features(
        split_933["X_train"],
        split_933["X_val"],
        split_933["X_test"],
        split_933["y_train"],
        split_933["y_val"],
        split_933["y_test"]
    )
    np.save("meta2/scenarios/embeddings_scenario_a_933.npy", scenario_a_933)

    # Para o split 60/20/20 com PCA
    scenario_b_602020 = create_scenario_pca_90(
        split_602020["X_train"],
        split_602020["X_val"],
        split_602020["X_test"],
        split_602020["y_train"],
        split_602020["y_val"],
        split_602020["y_test"]
    )
    np.save("meta2/scenarios/embeddings_scenario_b_602020_n_components.npy", scenario_b_602020)


    # Para o split 9/3/3 com PCA
    scenario_b_933 = create_scenario_pca_90(
        split_933["X_train"],
        split_933["X_val"],
        split_933["X_test"],
        split_933["y_train"],
        split_933["y_val"],
        split_933["y_test"]
    )
    np.save("meta2/scenarios/embeddings_scenario_b_933_n_components.npy", scenario_b_933)
    

    # Para o split 60/20/20 com ReliefF
    scenario_c_602020 = create_scenario_relieff_top15(
        split_602020["X_train"],
        split_602020["X_val"],
        split_602020["X_test"],
        split_602020["y_train"],
        split_602020["y_val"],
        split_602020["y_test"]
    )
    np.save("meta2/scenarios/embeddings_scenario_c_602020_top15_idx.npy", scenario_c_602020)

    np.save("meta2/scenario_c_602020_top15_idx.npy", scenario_c_602020["top15_idx"])

    # Para o split 9/3/3 com ReliefF
    scenario_c_933 = create_scenario_relieff_top15(
        split_933["X_train"],
        split_933["X_val"],
        split_933["X_test"],
        split_933["y_train"],
        split_933["y_val"],
        split_933["y_test"]
    )
    np.save("meta2/scenarios/embeddings_scenario_c_933_top15_idx.npy", scenario_c_933)
