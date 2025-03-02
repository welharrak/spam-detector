import numpy as np
from models.logistic_regression import sigmoid

#Function that predicts values given set, weights and bias
def predict_log_reg(X, weights, bias):
    z = np.dot(X,weights) + bias
    y = sigmoid(z)
    return np.round(y)

