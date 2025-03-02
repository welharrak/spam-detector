import numpy as np

# Sigmoid function
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Binary cross-entropy error function
def binary_cross_entropy(y,pred):
    N = len(y)
    return -1/N * np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))

# Logistic regression function
def logistic_regression(X, y, runs = 1000, lr = 0.01):
    m,n = X.shape
    weights = np.zeros(n)
    bias = 0

    for r in range(0,runs):
        # predict 
        z = np.dot(X, weights) + bias

        pred = sigmoid(z)
        error = pred - y

        # apply gradient
        weights -= lr * (np.dot(X.T, error)) / m
        bias -= lr * (np.sum(error)) / m

        #print loss each 100 steps
        if (r%100 == 0):
            print(f"Step {r+1}/{runs}, Loss: {binary_cross_entropy(y, pred):.4f}")

    return weights, bias




    