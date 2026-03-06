import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.module4_feature_statistics import extrairFeaturesJanela
from scripts.module4_pca_relief import normalizar_e_PCA_e_variancia
from src.har.classifier import knn_classifier




import numpy as np
from sklearn.decomposition import PCA


def predict(data):

    if data.shape != (256,9):
        print(f"Data of wrong shape. Received shape: {data.shape}")
        return

    # Load model

    model = np.load('results/plots/features_scenario_b_602020_n_components.npy', allow_pickle= True).item()

    x_train = model["X_train"]
    y_train = model["y_train"]
    x_val = model["X_val"]
    y_val = model["y_val"]
    pca = model["pca"]

    mean = model["train_mean"]
    std = model["train_std"]

    x_full = np.concatenate((x_train,x_val))
    y_full = np.concatenate((y_train,y_val))

    # Add column at index 0 (extrairFeaturesJanela expects the sensor data to be in columns 1-9, the values dont matter, the function doesnt access them)
    column = np.zeros((256,1))


    data = np.concatenate((column,data), axis = 1)

    # Feature extraction
    features = extrairFeaturesJanela(data)

    # Normalization
    features_normalized = (features - mean) / std
    features_normalized = features_normalized.reshape(1,-1)


    # PCA transformation
    features_pca = pca.transform(features_normalized)

    # Classification
    prediction = knn_classifier(training_data=x_full, training_classes=y_full, test_data=features_pca, k=1)

    return int(prediction[0])