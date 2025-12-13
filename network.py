import numpy as np
import random


class neuronNetwork:
    def __init__(self, architect: list[int]):
        self.architect = architect
        self.weights = []
        self.biases = []
        self.make_matrix(architect)
        
    def load_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def save_weights(self):
        return self.weights, self.biases


    def makeLayers(self, weightRange):
        i =0
        for layer in self.architect:
            if layer <= 0: #Number of neurons in a layer must be greater than 0
                raise ValueError("Number of neurons in a layer must be greater than 0")
            if i:
                n,spam = weightRange #Unpacking how many inputs neuron gets and it's weight range
                n = self.architect[i-1]+1 #Number of inputs for next layer is number of neurons in previous layer + 1 (bias)
                weightRange = n,spam
            self.layers.append(Layer([Neuron(weightRange, i, index) for index in range(layer)]))
            i+=1

    
    def make_matrix(self, architect: list[int]):
        layers = []
        biases = []
        # Assuming input size of the dataset is 3 (based on your teacher.py)
        in_dim = 3 
        
        for out_dim in architect:
            # Weights: (Current Layer Neurons, Previous Layer Neurons)
            weights = np.random.uniform(-1, 1, (out_dim, in_dim))
            # Biases: (Current Layer Neurons, 1)
            bias = np.random.uniform(-1, 1, (out_dim, 1))
            
            layers.append(weights)
            biases.append(bias)
            in_dim = out_dim
            
        self.weights = layers
        self.biases = biases
        

    '''
    def activate(self, params: list[int, float], get_all_outputs=False) -> float:
        overall_outputs = []
        input = [params for _ in range(len(self.layers[0].neurons))]  # Initialize input for the first layer
        
        overall_outputs.append(params)
        for i, layer in enumerate(self.layers):
            input = layer.activate(input)
            overall_outputs.append(input)
            #print(input)
            if i < len(self.layers) - 1:  # If not the last layer, prepare input for the next layer
                input = [input for _ in range(len(self.layers[i+1].neurons))]
            
        if get_all_outputs:
            return overall_outputs
        return input[0] if input else 0.0  # Return the output of the last layer, or 0 if empty'''
    

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
                error.append(layer_output - true_output)
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