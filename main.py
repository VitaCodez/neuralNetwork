import random
import time
import math
from network import neuronNetwork
from teacher_from_dataSet import Teacher
from dataHandler import DataHandler
     
    
def makeDataSet(n: int = 1000) -> list[list[int, float]]:
    dataSet = []
    for _ in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = random.uniform(-1, 1)
        dataSet.append([x1, x2, x3])
    return dataSet


def main():
    start = time.time()
    #dataSet = makeDataSet(1000)
    handler = DataHandler()
    trainingSet, correct_train = handler.get_training_data()
    testSet, correct_test = handler.get_testing_data()

    
    ARCHITECTURE = [34,16,1]
    network = neuronNetwork(ARCHITECTURE, (4, 0.1))
    teacher = Teacher(network, trainingSet, testSet, correct_train, correct_test, ARCHITECTURE)
    
    
    print("Initial Weights:", network.layers[-1].neurons[0].weights)
    teacher.Fit(200)
    print("Trained Weights:", network.layers[-1].neurons[0].weights)
    print(teacher.Test())
    
    for _ in range(10):
        sample = random.choice(teacher.testSet)
        n = random.randint(0, len(teacher.trainingSet) - 1)
        true = teacher.Y(n)
        pred = network.activate(teacher.trainingSet[n])
        print(f"Predikce: {pred:.3f} vs. Správná hodnota: {true:.3f}")
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

main()


