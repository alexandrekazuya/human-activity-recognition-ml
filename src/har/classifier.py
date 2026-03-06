import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter

def  euclidian_distance(point, array):
    ''' Returns distances between point and all points in array'''
    return np.sqrt(np.sum((point-array)**2, axis = 1))

def knn_classifier(training_data, training_classes, test_data, k):

    predictions = []

    for point in test_data:
        distances = euclidian_distance(point, training_data)

        # Get k indexes of sorted distances
        idxs = np.argsort(distances)[:k]

        knn_classes = training_classes[idxs]

        # Assign test data to most common label in knn
        counter = Counter(knn_classes)
        most_common = counter.most_common(1)
        predictions.append(most_common[0][0])

    return np.array(predictions)

def confusion_matrix(prediction_labels, true_labels):
    all_labels = np.unique(np.concatenate((prediction_labels, true_labels)))
    all_labels.sort()

    n_labels = len(all_labels)
    matrix = np.zeros((n_labels, n_labels), dtype = int)

    for pred_label, true_label in zip(prediction_labels, true_labels):
        pred_idx = np.where(all_labels == pred_label)[0][0]
        true_idx = np.where(all_labels == true_label)[0][0]

        matrix[true_idx][pred_idx] += 1

    return matrix, all_labels

def calculate_accuracy(prediction_labels, true_labels):

    return np.sum(prediction_labels == true_labels) / len(true_labels)


def class_precision(confusion_matrix, label_idx):
    tp = confusion_matrix[label_idx, label_idx]

    fp = np.sum(confusion_matrix[:,label_idx]) - tp

    if tp + fp == 0:
        return 0

    return tp/(tp + fp)

def class_recall(confusion_matrix, label_idx):
    tp = confusion_matrix[label_idx, label_idx]

    fn = np.sum(confusion_matrix[label_idx,:]) - tp

    if tp + fn == 0:
        return 0

    return tp/(tp + fn)

def f1_score(precision, recall):

    if precision + recall == 0:
        return 0

    return (2 * precision * recall) / (precision + recall)


def classification_metrics(prediction_labels, true_labels):

    conf_matrix, all_labels = confusion_matrix(prediction_labels, true_labels)
    n_labels = len(all_labels)
    all_labels.sort()

    accuracy = calculate_accuracy(prediction_labels, true_labels)

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(n_labels):
        prec = class_precision(conf_matrix, i)
        rec = class_recall(conf_matrix, i)
        f1 = f1_score(prec, rec)

        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)

    precision_mean = np.mean(precision_per_class)
    recall_mean = np.mean(recall_per_class)
    f1_mean = np.mean(f1_per_class)

    metrics = {
        'confusion_matrix': conf_matrix,
        'labels': all_labels,
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'precision_mean': precision_mean,
        'recall_mean': recall_mean,
        'f1_mean': f1_mean
    }

    return metrics

def print_metrics(metrics, labels):

    # Confusion matrix
    print("Confusion matrix")

    for i in range(len(labels)):
        row = str(labels[i]) #so mudei isto de labels[i] para str(labels[i])
        for cell in metrics['confusion_matrix'][i]:
            row += f"{cell:^10}"
        print(row)

    # Metrc for all labels
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision mean: {metrics['precision_mean']}")
    print(f"Recall mean: {metrics['recall_mean']}")
    print(f"f1 mean: {metrics['f1_mean']}")


    # Metrics per label
    print("\n Per class metrics:")
    for i in range(len(labels)):
        print(f"{labels[i]}:")
        print(f"Precision: {metrics['precision_per_class'][i]}")
        print(f"Recall: {metrics['recall_per_class'][i]}")
        print(f"f1: {metrics['f1_per_class'][i]}")

    return