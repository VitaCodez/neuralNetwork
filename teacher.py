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
                # Reshape labels based on task
                if batch_true.ndim == 1:
                    batch_true = batch_true.reshape(1, -1) # Regression: (Batch) -> (1, Batch)
                else:
                    batch_true = batch_true.T # Classification: (Batch, Classes) -> (Classes, Batch)
                activations, without_activation_f = self.network.activate(batch_input)
                gradients, errors = self.network.backpropagate(activations, without_activation_f, batch_true)
                errors = [np.sum(e, axis=1, keepdims=True) / self.batchSize for e in errors]
                gradients = [g / self.batchSize for g in gradients]
                self.network.update_weights(gradients, self.learningRate)
                self.network.update_biases(errors, self.learningRate)

    def test(self):
        self.testSet = np.array(self.testSet).T
        self.correct_test = np.array(self.correct_test)
        if self.network.architect[-1] != 1: # Classification
            self.correct_test = self.correct_test.T
            activations, without_activation_f = self.network.activate(self.testSet)
            pred = np.argmax(activations[-1], axis=0)
            true = np.argmax(self.correct_test, axis=0)
            accuracy = np.mean(pred == true)
            return accuracy
        else:  # Regression
            if self.correct_test.ndim == 1: # Reshape 1D array to (1, N) to match network output
                self.correct_test = self.correct_test.reshape(1, -1)
            else:
                self.correct_test = self.correct_test.T
            activations, without_activation_f = self.network.activate(self.testSet)
            pred = activations[-1]
            true = self.correct_test
            MSE = np.mean((pred - true)**2)
            return MSE