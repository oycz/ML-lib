import numpy as np


class Classifier:

    def __init__(self, data, training_method, classifying_method):
        self.training_method = training_method
        self.classifying_method = classifying_method
        self.data = data
        self.__train__()

    def __train__(self):
        train_result = self.method.train(self.data)
        self.W = train_result.W
        self.B = train_result.B

    def classify(self, batch_data):
        result = np.matmul(batch_data, self.W)
        np.add(result, self.B)
        return self.classifying_method(result)

