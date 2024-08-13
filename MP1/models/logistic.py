"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n_samples, n_features = X_train.shape
        self.w = np.random.rand(n_features + 1)
        X_train = np.concatenate((X_train, np.ones((n_samples, 1))), axis=1)
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X_train):
                y_pred = self.sigmoid(x_i @ self.w)
                self.w -= self.lr * (y_pred - y_train[idx]) * x_i
        
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]
        scores = X_test @ self.w
        y_pred = np.where(self.sigmoid(scores) > self.threshold, 1, 0)
        return y_pred
        
