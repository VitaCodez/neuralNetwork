import math

class Neuron: 
    def __init__(self, weights: list):
        self.weights = weights  
        self.error = 0.0
        self.output = 0.0  # Initialize output

    def activate(self, params: list[int, float]) -> float:
        z = sum(param * weight  for param, weight in zip(params, self.weights[:-1])) + self.weights[-1]  # Last weight is the bias
        self.output = math.tanh(z)  # Activation function
        return self.output  # Activation function 