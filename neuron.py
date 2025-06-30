import math
import random

class Neuron: 
    def __init__(self, weights: list):
        self.set_weights(weights) 
        self.error = 0.0
        self.output = 0.0  # Initialize output

    def set_weights(self, weights):
        t = type(weights) 
        if t == list:
            self.weights = weights
        elif t == tuple:
            n, spam = weights
            self.weights = [random.uniform(-spam, spam) for _ in range(n)]
        else:
            raise ValueError(f"Unsuportet type for Neuron.weights{t}")
            
            
        

    def activate(self, params: list[int, float]) -> float:
        z = sum(param * weight  for param, weight in zip(params, self.weights[:-1])) + self.weights[-1]  # Last weight is the bias
        self.output = math.tanh(z)  # Activation function
        return self.output  # Activation function 