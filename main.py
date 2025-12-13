import time
import math
from network import neuronNetwork
from teacher import Teacher
import numpy as np
    
def makeDataSet(n: int = 1000) -> np.ndarray:
    dataSet = np.random.uniform(-1, 1, (n, 3))
    return dataSet

def makeCorrectSet(dataSet: np.ndarray) -> np.ndarray:
    correctSet = []
    for i in range(len(dataSet)):
        # y = sin(3*x1) + x2^2 * cos(3*x3)
        val = np.sin(3 * dataSet[i][0]) + (dataSet[i][1]**2 * np.cos(3 * dataSet[i][2]))
        correctSet.append(val)
    return correctSet 

def main():
    start = time.time()
    trainingSet = makeDataSet(2000)
    correct_train = makeCorrectSet(trainingSet)

    testSet = makeDataSet(200)
    correct_test = makeCorrectSet(testSet)

    # Deeper architecture for non-linear problem
    ARCHITECTURE = [32, 16, 1]
    network = neuronNetwork(ARCHITECTURE)
    teacher = Teacher(network, trainingSet, correct_train, testSet, correct_test)

    EPOCHS = 200
    print(f"Strating training for {EPOCHS} epochs...")
    teacher.fit(EPOCHS)
    
    # SME is likely a 1-element array, use .item() to get scalar
    SME = teacher.test()
    if isinstance(SME, np.ndarray):
        SME = SME.item()
        
    print(f"SME: {SME:.6f}, RMSE: {math.sqrt(SME):.6f}")
    
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    
main()


