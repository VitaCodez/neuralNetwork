import math
import random
from network import neuronNetwork
from neuron import Neuron
PI2 = math.pi * 2  

class Teacher:
    def __init__(self, network, trainingSet, testSet, correct_train, correct_test, architecture: list):
        self.network = network
        self.learningRate = 1e-2  # Learning rate
        self.trainingSet = trainingSet
        self.testSet = testSet
        self.training_correct = correct_train
        self.correct_test = correct_test
        self.correct_network = neuronNetwork(architecture, (4, 1))  # Assuming the last layer has 1 neuron for output
        

    def Y(self, index) -> float:
        return self.training_correct[index]
        
          
    def Loss(self, params: list[int, float], index) -> float:
        return 0.5 * (self.Y(index) - self.network.activate(params))**2
    
    def LossDerivative(self, params: list[int, float], index) -> list[float]:
        y = self.Y(index)
        output = self.network.activate(params)
        
        return (output - y)

    def Backpropagation(self, params: list[int, float], index):
        gradientMap = []
        ouput_of_layers = self.network.get_ouput_of_layers(params)
        for i, layer in enumerate(reversed(self.network.layers)):
            gradientMap.append([])
            for neuron in layer.neurons:
                g = self.Gradient(params, neuron, ouput_of_layers, index)
                gradientMap[i].append(g)
        return gradientMap
    
    
    def Gradient(self, params: list[int, float], neuron: Neuron, ouput_of_layers, index_cor) -> list[float]:

        if neuron in self.network.layers[-1].neurons:
            # Calculate error for the output layer
            error = self.LossDerivative(params, index_cor)
            layer_index = len(self.network.layers)-1
        else:
            # Calculate error for hidden layers
            error = 0.0
            layer_index, index = neuron.pos
            next_layer = self.network.layers[layer_index + 1]
            for next_neuron in next_layer.neurons:
                error += next_neuron.error * next_neuron.weights[index]

        # Derivative of tanh(x) = 1 - tanh(x)^2
        activation_derivative = 1.0 - neuron.output ** 2
        neuron.error = error * activation_derivative

        neuron_inputs = ouput_of_layers[layer_index]
        
        gradient = [neuron.error * X for X in neuron_inputs]
        gradient.append(neuron.error)  # Bias gradient
        return gradient


    def UpdateWeights(self, params: list[int, float], index) -> list[float]: 
        gradientMap = self.Backpropagation(params, index)

        # Aktualizace vah v síti
        for i, layer in enumerate(reversed(self.network.layers)):
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= self.learningRate * gradientMap[i][j][k]
                    
    
    def Test(self):
        results = []
        for index, param in enumerate(self.testSet):
            results.append((self.correct_test[index] - self.network.activate(param)) ** 2)
        mean = sum(results) / len(results)
        return f"Squared Mean Error: {mean:.4f}"


    def Fit(self, iterations: int):
        epoch = 1
        for _ in range(iterations):
            for index in range(len(self.trainingSet)):
                params = self.trainingSet[index]
                    
                self.UpdateWeights(params, index)
                epoch += 1

