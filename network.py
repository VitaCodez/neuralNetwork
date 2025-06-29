from layer import Layer
from neuron import Neuron
import random


class neuronNetwork:
    def __init__(self, architect: list[int], weightRange):
        self.layers = []
        self.architect = architect
        self.weightRange = weightRange
        self.makeLayers()
        

    def makeLayers(self):
        for i in self.architect:
            if i <= 0:
                raise ValueError("Number of neurons in a layer must be greater than 0")
            self.layers.append(Layer([Neuron([random.uniform(-1*self.weightRange, 1*self.weightRange) for _ in range(4)]) for _ in range(i)]))

    def activate(self, params: list[int, float]) -> float:
        input = [params for _ in range(len(self.layers[0].neurons))]  # Initialize input for the first layer
        
        
        for i, layer in enumerate(self.layers):
            input = layer.activate(input)
            #print(input)
            if i < len(self.layers) - 1:  # If not the last layer, prepare input for the next layer
                input = [input for _ in range(len(self.layers[i+1].neurons))]
            
        
        return input[0] if input else 0.0  # Return the output of the last layer, or 0 if empty