import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from meta2.splitFeatures import create_scenario_pca_90
from src.har.classifier import knn_classifier, calculate_accuracy

def load_data():
    print("Loading data...")
    #load embeddings
    X_emb = np.load("data/embeddings.npy")
    y_emb = np.load("data/y.npy")
    part_emb = np.load("data/participants.npy")
    
    #load features
    X_feat = np.load("data/features.npy")
    y_feat = np.load("data/y_features.npy")
    part_feat = np.load("data/participants_features.npy")
    
    return (X_emb, y_emb, part_emb), (X_feat, y_feat, part_feat)

def run_iteration(X_emb, y_emb, part_emb, X_feat, y_feat, part_feat, seed):
    #split embeddings (60-20-20)
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_emb, y_emb, test_size=0.2, random_state=seed, stratify=y_emb
    )
    X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
        X_train_e, y_train_e, test_size=0.25, random_state=seed, stratify=y_train_e
    )
    
    #split features (60-20-20)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_feat, y_feat, test_size=0.2, random_state=seed, stratify=y_feat
    )
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_train_f, y_train_f, test_size=0.25, random_state=seed, stratify=y_train_f
    )

    #model 1: EmbeddingsScenario A (60-20-20) all
    #Best k foi 1
    X_full_e = np.concatenate((X_train_e, X_val_e))
    y_full_e = np.concatenate((y_train_e, y_val_e))
    preds_e = knn_classifier(X_full_e, y_full_e, X_test_e, k=1)
    acc_e = calculate_accuracy(preds_e, y_test_e)

    #model 2: Features Scenario A (60-20-20) all
    #Best k foi 1
    X_full_f = np.concatenate((X_train_f, X_val_f))
    y_full_f = np.concatenate((y_train_f, y_val_f))
    preds_f = knn_classifier(X_full_f, y_full_f, X_test_f, k=1)
    acc_f = calculate_accuracy(preds_f, y_test_f)

    #model 3, melhor modelo: Features Scenario B (60-20-20) pca
    #Best k foi 1
    scenario_pca = create_scenario_pca_90(
        X_train_f, X_val_f, X_test_f, y_train_f, y_val_f, y_test_f
    )
    X_train_pca = scenario_pca["X_train"]
    X_val_pca = scenario_pca["X_val"]
    X_test_pca = scenario_pca["X_test"]
    
    X_full_pca = np.concatenate((X_train_pca, X_val_pca))
    preds_pca = knn_classifier(X_full_pca, y_full_f, X_test_pca, k=1)
    acc_pca = calculate_accuracy(preds_pca, y_test_f)
    
    return acc_e, acc_f, acc_pca

def main():
    (X_emb, y_emb, part_emb), (X_feat, y_feat, part_feat) = load_data()
    
    n_iterations = 10
    results_e = []
    results_f = []
    results_pca = []
    
    print(f"\nRunning 10 iterations...")
    
    for i in range(n_iterations):
        seed = 42 + i #seed diferente para cada iteracao
        acc_e, acc_f, acc_pca = run_iteration(
            X_emb, y_emb, part_emb, X_feat, y_feat, part_feat, seed
        )
        results_e.append(acc_e)
        results_f.append(acc_f)
        results_pca.append(acc_pca)
        print(f"Iteracao {i+1}: Embeddings={acc_e:.4f}, Features={acc_f:.4f}, PCA={acc_pca:.4f}")

    print("\n------Results---")
    print(f"Embeddings (All): Mean={np.mean(results_e):.4f}, Std={np.std(results_e):.4f}")
    print(f"Features (All):   Mean={np.mean(results_f):.4f}, Std={np.std(results_f):.4f}")
    print(f"Features (PCA):   Mean={np.mean(results_pca):.4f}, Std={np.std(results_pca):.4f}")
    
    print("\n-----Hypothesis testing (vs Features (PCA))--")
    
    #PCA vs embeddings all
    print("\ncomparacao: Features (PCA) vs Embeddings (All)")

    stat, p = stats.ttest_rel(results_pca, results_e)
    print(f"Paired ttest: p-value = {p:.4e}")
    print(f"conclusao: {'Significant' if p < 0.05 else 'Not Significant'}")

    #PCA vs features all
    print("\ncomparacao: Features (PCA) vs Features (All)")
    stat, p = stats.ttest_rel(results_pca, results_f)
    print(f"Paired ttest: p-value = {p:.4e}")
    print(f"conclusao: {'Significant' if p < 0.05 else 'Not Significant'}")

if __name__ == "__main__":
    main()
