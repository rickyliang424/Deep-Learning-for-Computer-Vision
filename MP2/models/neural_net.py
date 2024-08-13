"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i -1 ])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        
        for i in range(1, num_layers + 1):
            self.params['mw'+str(i)] = np.zeros(self.params[f"W{i}"].shape)
            self.params['vw'+str(i)] = np.zeros(self.params[f"W{i}"].shape)
            self.params['mb'+str(i)] = np.zeros(self.params[f"b{i}"].shape)
            self.params['vb'+str(i)] = np.zeros(self.params[f"b{i}"].shape)

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        

        return np.dot(X, W) + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        de_dw = np.dot(X.T, de_dz) / N + reg * W
       
        de_db = np.sum(de_dz, axis=0) / N
       
        de_dx = np.dot(de_dz, W.T)

        return de_dw, de_db, de_dx
        

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return X * (1 - X)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean((np.square(y - p)))
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return -2 * (y - p) / y.shape[0]
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return -2 * (y - p) * p * (1 - p) / y.shape[0]
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        input = X
        self.outputs['layer0'] = X  
        for i in range(1, self.num_layers):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            input = self.linear(W, input, b)
            if i < self.num_layers - 1:  
                input = self.relu(input)
            self.outputs['layer' + str(i)] = input
       
        final_W, final_b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        input = self.linear(final_W, input, final_b)
        input = self.sigmoid(input)
        self.outputs['layer' + str(self.num_layers)] = input
        final_output = input

        return final_output


    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        # Compute the gradient at the last layer
        final_output = self.outputs['layer' + str(self.num_layers)]
        mse_loss = self.mse(y, final_output)
        mse_grad = self.mse_sigmoid_grad(y, final_output)
      
        before_sigmoid_grad = mse_grad
        xi_gradients = {}
        xi_gradients["X" + str(self.num_layers)] = before_sigmoid_grad
        for i in range(self.num_layers, 0, -1):
            # Calculate gradients for layer i parameters
            if i == 1:  # For the first layer, use the original input
                X_input = self.outputs["layer0"]
            else:  # For other layers, use the output of the previous layer
                X_input = self.outputs["layer" + str(i - 1)]
            
            wi_grad, bi_grad, xi_grad = self.linear_grad(
                self.params["W" + str(i)], X_input, self.params["b" + str(i)], 
                xi_gradients["X" + str(i)], reg=0, N=y.shape[0]
            )
            self.gradients["W" + str(i)] = wi_grad
            self.gradients["b" + str(i)] = bi_grad

            # Calculate the gradient for output from the previous layer
            if i > 1:  # No need to calculate xi_grad for the input layer
                xi_gradients["X" + str(i - 1)] = xi_grad * self.relu_grad(self.outputs["layer" + str(i - 1)])

        return mse_loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt=="SGD":
          #print('sgd')
          for i in range(1,self.num_layers+1):
            self.params[f'W{i}'] = self.params[f'W{i}'] - lr * self.gradients[f'W{i}']
            self.params[f'b{i}'] = self.params[f'b{i}'] - lr * self.gradients[f'b{i}']
        elif opt=="Adam":
          #print('adam')
          for i in range(1,self.num_layers+1):
            self.params[f'mw{i}'] = b1 * self.params[f'mw{i}'] + \
            (1 - b1) * self.gradients[f'W{i}']
            self.params[f'vw{i}'] = b2 *  self.params[f'vw{i}'] + \
            (1 - b2) * (self.gradients[f'W{i}'] ** 2)
            self.params[f'W{i}'] = self.params[f'W{i}'] - \
            (lr / (np.sqrt(self.params[f'vw{i}'])+eps) )* self.params[f'mw{i}'] 

            self.params[f'mb{i}'] = b1 * self.params[f'mb{i}'] + \
            (1 - b1) * self.gradients[f'b{i}']
            self.params[f'vb{i}'] = b2 *  self.params[f'vb{i}'] + \
            (1 - b2) * (self.gradients[f'b{i}'] ** 2)
            self.params[f'b{i}'] = self.params[f'b{i}'] - \
            (lr / (np.sqrt(self.params[f'vb{i}'])+eps) )* self.params[f'mb{i}'] 
        