import time
import math
from network import neuronNetwork
from teacher import Teacher
import numpy as np
    
def makeDataSet(n: int = 1000) -> np.ndarray:
    dataSet = np.random.uniform(-1.5, 1.5, (n, 4))
    return dataSet

def makeCorrectSet(dataSet: np.ndarray) -> np.ndarray:
    correctSet = []
    for i in range(len(dataSet)):
        # HARD CORE: The "Onion of Doom" (Modulo Spherical Shells)
        # Classes repeat: 0 -> 1 -> 2 -> 0 -> 1 -> 2 ...
        # This breaks monotonicity. The network can't just say "Big Inputs = Class 2".
        
        r_squared = np.sum(dataSet[i]**2)
        radius = np.sqrt(r_squared)
        
        # Scale radius so we get multiple bands within the range [0, 3.0]
        # int(radius * 2.0) breaks space into 0.5-width shells
        band_index = int(radius * 2.0)
        
        class_id = band_index % 3
        
        if class_id == 0:
            correctSet.append([1, 0, 0])
        elif class_id == 1:
            correctSet.append([0, 1, 0])
        else:
            correctSet.append([0, 0, 1])
            
    return np.array(correctSet) 

def main():
    start = time.time()
    trainingSet = makeDataSet(20000)
    correct_train = makeCorrectSet(trainingSet)

    testSet = makeDataSet(200)
    correct_test = makeCorrectSet(testSet)

    # Deeper architecture for non-linear problem
    ARCHITECTURE = [32, 16, 3]
    network = neuronNetwork(ARCHITECTURE, 4)
    teacher = Teacher(network, trainingSet, correct_train, testSet, correct_test)

    EPOCHS = 500
    print(f"Strating training for {EPOCHS} epochs...")
    teacher.fit(EPOCHS)
    
    # SME is likely a 1-element array, use .item() to get scalar
    
    if ARCHITECTURE[-1] == 1:
        SME = teacher.test()
        if isinstance(SME, np.ndarray):
            SME = SME.item()
        
        print(f"SME: {SME:.6f}, RMSE: {math.sqrt(SME):.6f}")
    else:
        accuracy = teacher.test()
        print(f"Accuracy: {accuracy * 100:.2f}%")
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    
main()


