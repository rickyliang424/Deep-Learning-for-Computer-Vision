"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = j means that X[i] has label j, where 0 <= j < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        n_sample, n_feature = X_train.shape
        batch_w = np.zeros((n_feature, self.n_class))
        
        for i in range(n_sample):
            for j in range(self.n_class):
                if j != y_train[i]:
                    if np.dot(np.transpose(self.w)[y_train[i]], X_train[i]) - np.dot(np.transpose(self.w)[j], X_train[i]) < 1:
                        np.transpose(batch_w)[y_train[i]] = np.transpose(batch_w)[y_train[i]] + self.lr * X_train[i]
                        np.transpose(batch_w)[j] = np.transpose(batch_w)[j] - self.lr * X_train[i]
                np.transpose(batch_w)[j] = np.transpose(batch_w)[j] - (self.lr * self.reg_const / n_sample) * np.transpose(self.w)[j]
        return batch_w / n_sample

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labe
            s
        """
        # TODO: implement me
        # Parse dimensions
        n_sample, n_feature = X_train.shape
        self.w = np.random.rand(n_feature, self.n_class)
        batch_size = 1024
        batches = n_sample // batch_size
        indices = np.arange(n_sample)

        for epoch in range(self.epochs):
            y_pred = X_train @ self.w
            temp = [np.argmax(i) for i in y_pred]
            np.random.shuffle(indices)
            for batch in range(batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                if end >= n_sample:
                    end = -1
                batch_indices = indices[start: end]
                batch_w = self.calc_gradient(X_train[batch_indices], y_train[batch_indices])
                self.w += self.lr * batch_w
            
            self.lr *= 0.85


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
        y_pred = X_test @ self.w
        return [np.argmax(i) for i in y_pred]