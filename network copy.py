from layer import Layer
from neuron import Neuron
import random


class neuronNetwork:
    def __init__(self, architect: list[int], weightRange):
        self.layers = []
        self.architect = architect
        self.makeLayers(weightRange)
        

    def makeLayers(self, weightRange):
        i =0
        for neurons in self.architect:
            if neurons <= 0:
                raise ValueError("Number of neurons in a layer must be greater than 0")
            if i:
                n,spam = weightRange
                n = self.architect[i-1]+1
                weightRange = n,spam
            self.layers.append(Layer([Neuron(weightRange) for _ in range(neurons)]))
            i+=1

    def activate(self, params: list[int, float], get_all_outputs=False) -> float:
        overall_outputs = []
        input = [params for _ in range(len(self.layers[0].neurons))]  # Initialize input for the first layer
        
        overall_outputs.append(input)
        for i, layer in enumerate(self.layers):
            input = layer.activate(input)
            overall_outputs.append(input)
            #print(input)
            if i < len(self.layers) - 1:  # If not the last layer, prepare input for the next layer
                input = [input for _ in range(len(self.layers[i+1].neurons))]
            
        if get_all_outputs:
            return overall_outputs
        return input[0] if input else 0.0  # Return the output of the last layer, or 0 if empty
    
    def get_ouput_of_layers(self, params):
        return self.activate(params, get_all_outputs=True)