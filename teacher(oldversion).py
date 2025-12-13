import math
import random
from network import neuronNetwork
from neuron import Neuron
PI2 = math.pi * 2  

class Teacher:
    def __init__(self, network, dataSet: list[list[int, float]], correctWeights, architecture: list):
        self.network = network
        self.batchSize = 32
        self.learningRate = 1e-2  # Learning rate
        self.correctWeights = correctWeights
        self.trainingSet = []
        self.testSet = []
        self.splitDataSet(dataSet)
        self.correct_network = neuronNetwork(architecture, (4, 1))  # Assuming the last layer has 1 neuron for output
        
    def splitDataSet(self, dataSet):
        split_ratio = 0.8  #ratio in which will be dataSet/testSet
        length  = len(dataSet)
        self.trainingSet = dataSet[:int(length * split_ratio)]
        self.testSet = dataSet[int(split_ratio * length):]

    def Y(self, params:list[int, float]) -> float:
        
        
        x1, x2, x3 = params
        
        #return math.tanh(x1  * x2 * x3)  
        return math.sin(x1 * PI2)  
        #return self.correct_network.activate(params)  
        
    
    def Loss(self, params: list[int, float]) -> float:
        return 0.5 * (self.Y(params) - self.network.activate(params))**2
    
    def LossDerivative(self, params: list[int, float]) -> list[float]:
        y = self.Y(params)
        output = self.network.activate(params)
        
        return (output - y)

    def Backpropagation(self, params: list[int, float]):
        gradientMap = []
        ouput_of_layers = self.network.get_ouput_of_layers(params)
        for i, layer in enumerate(reversed(self.network.layers)):
            gradientMap.append([])
            for neuron in layer.neurons:
                g = self.Gradient(params, neuron, ouput_of_layers)
                gradientMap[i].append(g)
        return gradientMap
    
    
    def Gradient(self, params: list[int, float], neuron: Neuron, ouput_of_layers) -> list[float]:

        if neuron in self.network.layers[-1].neurons:
            # Calculate error for the output layer
            error = self.LossDerivative(params)
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


    def UpdateWeights(self, params: list[int, float]) -> list[float]: 
        gradientMap = self.Backpropagation(params)

        # Aktualizace vah v síti
        for i, layer in enumerate(reversed(self.network.layers)):
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= self.learningRate * gradientMap[i][j][k]
                    
    
    def Test(self):
        results = []
        for param in self.testSet:
            results.append(self.Loss(param)*2)
        mean = sum(results) / len(results)
        return f"Squared Mean Error: {mean:.4f}"


    def Fit(self, iterations: int):
        epoch = 1
        for i in range(iterations):
            for n in range(len(self.trainingSet)):
                params = self.trainingSet[n]
                
                if epoch % 10000 == 0:
                    sample = random.randint(10,len(self.trainingSet)-1)
                    print(self.quickTest(self.trainingSet[sample-10:sample]))
                self.UpdateWeights(params)
                epoch += 1

    def quickTest(self, data):
        error = []
        for param in data:
            true = self.Y(param)
            pred = self.network.activate(param)
            error.append((pred - true) ** 2)
        return sum(error) / len(error)