"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, batch_size: int, lr_decay_rate: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.batch_size = batch_size
        self.lr_decay_rate = lr_decay_rate

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        accuracy = []
        n_sample, n_feature = X_train.shape
        self.w = np.random.rand(self.n_class, n_feature)
        self.b = np.random.rand(self.n_class, 1)
        batches = self.batch_size * np.arange(n_sample // self.batch_size + 1)
        batches = np.append(batches, n_sample - 1).tolist()
        
        for epoch in range(self.epochs):
            idx = np.arange(X_train.shape[0])
            np.random.shuffle(idx)
            X_train = X_train[idx.tolist(),:]
            y_train = y_train[idx.tolist()]
            for batch in range(len(batches)-1):
                x_data = X_train[batches[batch]:batches[batch+1],:].T
                y_data = y_train[batches[batch]:batches[batch+1]]
                score = self.w @ x_data + self.b
                for i in range(score.shape[1]):
                    for j in range(score.shape[0]):
                        if score[j,i] > score[y_data[i],i] and j != y_data[i]:
                            self.w[y_data[i],:] += self.lr * x_data[:,i]
                            self.w[j,:] -= self.lr * x_data[:,i]
            self.lr = self.lr * self.lr_decay_rate

        ## plot training history
            y_pred = np.argmax(self.w @ X_train.T + self.b, axis=0)
            accuracy.append(np.sum(y_train == y_pred) / len(y_train) * 100)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(accuracy)
        plt.grid()
        
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
        y_pred = np.argmax(self.w @ X_test.T + self.b, axis=0)
        return y_pred
