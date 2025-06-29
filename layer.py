

class Layer:
    def __init__(self, neurons: list):
        self.neurons = neurons

    def activate(self, wholeParams: list[list[int, float]]) -> list[float]:
        layer_output = []
        for i, neuron in enumerate(self.neurons):
            neuron_output = neuron.activate(wholeParams[i])
            layer_output.append(neuron_output)
        return layer_output