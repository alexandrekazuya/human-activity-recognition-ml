import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from meta2.splitEmbeddings import split60_20_20, split933, create_scenario_all_features, create_scenario_pca_90, create_scenario_relieff_top15


def load_feature_arrays(features_path="meta2/features.npy", y_path="meta2/y_features.npy", participants_path="meta2/participants_features.npy"):
    X = np.load(features_path)
    y = np.load(y_path)
    participants = np.load(participants_path)
    return X, y, participants

if __name__ == '__main__':
    X, y, participants = load_feature_arrays()

    split_602020 = split60_20_20(X, y, participants)
    split_933 = split933(X, y, participants)

    #cenario a: all features
    scenario_a_602020 = create_scenario_all_features(
        split_602020["X_train"], split_602020["X_val"], split_602020["X_test"],
        split_602020["y_train"], split_602020["y_val"], split_602020["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_a_602020.npy", scenario_a_602020)

    scenario_a_933 = create_scenario_all_features(
        split_933["X_train"], split_933["X_val"], split_933["X_test"],
        split_933["y_train"], split_933["y_val"], split_933["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_a_933.npy", scenario_a_933)

    #cenario b: pca90
    scenario_b_602020 = create_scenario_pca_90(
        split_602020["X_train"], split_602020["X_val"], split_602020["X_test"],
        split_602020["y_train"], split_602020["y_val"], split_602020["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_b_602020_n_components.npy", scenario_b_602020)


    scenario_b_933 = create_scenario_pca_90(
        split_933["X_train"], split_933["X_val"], split_933["X_test"],
        split_933["y_train"], split_933["y_val"], split_933["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_b_933_n_components.npy", scenario_b_933)

    #cenario c: relieff top15
    scenario_c_602020 = create_scenario_relieff_top15(
        split_602020["X_train"], split_602020["X_val"], split_602020["X_test"], 
        split_602020["y_train"], split_602020["y_val"], split_602020["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_c_602020_top15_idx.npy", scenario_c_602020)

    scenario_c_933 = create_scenario_relieff_top15(
        split_933["X_train"], split_933["X_val"], split_933["X_test"], 
        split_933["y_train"], split_933["y_val"], split_933["y_test"]
    )
    np.save("meta2/scenarios/features_scenario_c_933_top15_idx.npy", scenario_c_933)
