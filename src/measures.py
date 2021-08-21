from sklearn.metrics import f1_score
import numpy as np
import torch


def f1_score_function(predictions, labels):
    predictions_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, predictions_flat, average='weighted')


def accuracy_per_class(predictions, labels, class_number):
    prediction_flat = torch.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    tt = 0
    for label in range(class_number):
        y_predictions = prediction_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        tt += (len(y_predictions[y_predictions == label]) + 1) / (len(y_true) + 1)
    return tt / class_number
