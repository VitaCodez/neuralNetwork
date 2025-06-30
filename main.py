import random
from .network import neuronNetwork
from .teacher import Teacher

            
    
def makeDataSet(n: int = 1000) -> list[list[int, float]]:
    dataSet = []
    for _ in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = random.uniform(-1, 1)
        dataSet.append([x1, x2, x3])
    return dataSet


def main():
    dataSet = makeDataSet(500)
    #weights = [random.uniform(-1, 1) for _ in range(4)] # 3 weights + 1 bias
    #network = Neuron(weights)
    network = neuronNetwork([4,4,4,1], (4, 0.1))
    teacher = Teacher(network, dataSet, [3.43, -442.323, 632.81, 31.43])
    
    
    print("Initial Weights:", network.layers[-1].neurons[0].weights)
    teacher.Fit(100)
    print("Trained Weights:", network.layers[-1].neurons[0].weights)
    print(teacher.Test())
    print(teacher.network.layers[1].neurons[0].weights)
    for _ in range(10):
        sample = random.choice(teacher.testSet)
        true = teacher.Y(sample)
        pred = network.activate(sample)
        print(f"Predikce: {pred:.3f} vs. Správná hodnota: {true:.3f}")

main()