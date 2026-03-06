import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src.har.classifier import knn_classifier, classification_metrics, print_metrics, calculate_accuracy

def evaluate_scenario_data(X_train, y_train, X_val, y_val, X_test, y_test, scenario_name):
    print(f"\nEvaluating: {scenario_name}")
    
    best_k = 1
    best_acc = 0.0
    
    k_values = [k for k in range(1, 20, 2)] #odd
    
    for k in k_values:
        preds_val = knn_classifier(X_train, y_train, X_val, k)
        acc = calculate_accuracy(preds_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            
    print(f"Best k: {best_k} (Accuracy: {best_acc})")
    
    #retrain with the dataset (train + validation)
    print("Training on Train + Val and evaluating on Test...")
    X_full = np.concatenate((X_train, X_val))
    y_full = np.concatenate((y_train, y_val))

    preds_test = knn_classifier(X_full, y_full, X_test, best_k)
    
    metrics = classification_metrics(preds_test, y_test)
    print_metrics(metrics, metrics['labels'])

if __name__ == "__main__":
    
    scenarios = [
        # Embeddings
        ("results/plots/embeddings_scenario_a_933.npy", "Embeddings - Scenario A (9-3-3)"),
        ("results/plots/embeddings_scenario_b_933_n_components.npy", "Embeddings - Scenario B (9-3-3)"),
        ("results/plots/embeddings_scenario_c_933_top15_idx.npy", "Embeddings - Scenario C (9-3-3)"),
        ("results/plots/embeddings_scenario_a_602020.npy", "Embeddings - Scenario A (60-20-20)"),
        ("results/plots/embeddings_scenario_b_602020_n_components.npy", "Embeddings - Scenario B (60-20-20)"),
        ("results/plots/embeddings_scenario_c_602020_top15_idx.npy", "Embeddings - Scenario C (60-20-20)"),
        
        # Features
        ("results/plots/features_scenario_a_933.npy", "Features - Scenario A (9-3-3)"),
        ("results/plots/features_scenario_b_933_n_components.npy", "Features - Scenario B (9-3-3)"),
        ("results/plots/features_scenario_c_933_top15_idx.npy", "Features - Scenario C (9-3-3)"),
        ("results/plots/features_scenario_a_602020.npy", "Features - Scenario A (60-20-20)"),
        ("results/plots/features_scenario_b_602020_n_components.npy", "Features - Scenario B (60-20-20)"),
        ("results/plots/features_scenario_c_602020_top15_idx.npy", "Features - Scenario C (60-20-20)"),
    ]

    for filename, name in scenarios:
        try:
            scenario_data = np.load(filename, allow_pickle=True)
            
            data = scenario_data.item()
            
            print(f"Loaded {name}. X_train shape: {data['X_train'].shape}")

            evaluate_scenario_data(
                data["X_train"], data["y_train"],
                data["X_val"], data["y_val"],
                data["X_test"], data["y_test"],
                name
            )
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")
