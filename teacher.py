import numpy as np
from network import neuronNetwork 

class Teacher:
    def __init__(self, network, trainingSet, correct_train, testSet, correct_test):
        self.network = network
        self.batchSize = 32
        self.learningRate = 1e-2  # Learning rate
        self.trainingSet = trainingSet
        self.correct_train = correct_train
        self.testSet = testSet
        self.correct_test = correct_test
        

    def get_batches(self):
        train_batches = []
        correct_train_batches = []
        for i in range(0, len(self.trainingSet)//self.batchSize *self.batchSize, self.batchSize):
            train_batches.append(self.trainingSet[i: i+self.batchSize]) #Split training set into batches
            correct_train_batches.append(self.correct_train[i: i+self.batchSize])
        return train_batches, correct_train_batches

    def fit(self, epochs):
        for _ in range(epochs):
            train_batches, correct_train_batches = self.get_batches()
            for batch_ind in range(len(train_batches)):
                batch_input = np.array(train_batches[batch_ind]).T
                batch_true = np.array(correct_train_batches[batch_ind])
                activations, without_activation_f = self.network.activate(batch_input)
                gradients, errors = self.network.backpropagate(activations, without_activation_f, batch_true)
                errors = [np.sum(e, axis=1, keepdims=True) / self.batchSize for e in errors]
                gradients = [g / self.batchSize for g in gradients]
                self.network.update_weights(gradients, self.learningRate)
                self.network.update_biases(errors, self.learningRate)
                


    def test(self):
        error = []
        for i in range(len(self.testSet)):
            sample = self.testSet[i]
            true_output = self.correct_test[i]
            activations, without_activation_f = self.network.activate(sample)
            error.append((true_output - activations[-1][0]) ** 2)
        return sum(error) / len(error)
            