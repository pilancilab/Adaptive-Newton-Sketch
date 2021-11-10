import numpy as np
from loss_funcs import safe_sigmoid

def predict_proba(features, weights):
    Ax = features.dot(weights)
    e = safe_sigmoid(Ax)
    probs = e > 0.5
    return probs

def predict_classes(features, weights):
    probs = predict_proba(features, weights)
    return probs > 0.5

def evaluate(features, weights, target):
    preds = predict_classes(features, weights)
    print('Accuracy: ', np.mean(target == preds), '\n')

def evaluate_acc(features, weights, target):
    preds = predict_classes(features, weights)
    return np.mean(target == preds)

