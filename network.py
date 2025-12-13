import numpy as np
import random
import math

class neuronNetwork:
    def __init__(self, architect: list[int], input_shape=3):
        self.architect = architect
        self.input_shape = input_shape
        self.weights = []
        self.biases = []
        self.make_matrix(architect)
        
        
    def load_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def save_weights(self):
        return self.weights, self.biases

    
    def make_matrix(self, architect: list[int]):
        layers = []
        biases = []

        in_dim = self.input_shape 
        limit = 1 / math.sqrt(in_dim)

        for out_dim in architect:
            # Weights: (Current Layer Neurons, Previous Layer Neurons)
            weights = np.random.uniform(-limit, limit, (out_dim, in_dim))
            # Biases: (Current Layer Neurons, 1)
            bias = np.zeros((out_dim, 1))
            
            layers.append(weights)
            biases.append(bias)
            in_dim = out_dim
            
        self.weights = layers
        self.biases = biases
        
    def activate(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        without_activation_f = [x]
        activations = []
        for i, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            out = weights @ x + bias
            without_activation_f.append(out)
            if i < len(self.weights) - 1: # Use Tanh for hidden layers, Linear for output layer
                out = np.tanh(out) # else: out = out (Linear)
            activations.append(out)
            x = out
        return activations, without_activation_f
                    
    def backpropagate(self, activations, without_activation_f, true_output) -> list[np.ndarray]:
        error = []
        gradients = []
        for i in reversed(range(len(self.weights))):
            layer_input = activations[i-1] if i > 0 else without_activation_f[0]
            layer_output = activations[i]
            if i == len(self.weights) - 1:
                error.append(layer_output - true_output.reshape(1, -1))
            else:
                error.append((self.weights[i+1].T @ error[-1]) * (1 - layer_output**2))
            gradients.append(error[-1] @ layer_input.T)
        return gradients[::-1], error[::-1]

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def update_biases(self, errors, learning_rate):
        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * errors[i]