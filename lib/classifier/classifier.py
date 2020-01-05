class Classifier:

    def __init__(self, X, Y, training_method, classifying_method, training_args, classifying_args):
        self.training_method = training_method
        self.training_args = training_args
        self.classifying_method = classifying_method
        self.classifying_args = classifying_args
        self.X = X
        self.Y = Y
        self.__train__()

    def __train__(self):
        training_result = self.training_method.train(self.X, self.Y, self.training_args)
        self.model = training_result

    def classify(self, batch_data):
        return self.classifying_method(batch_data, self.model, self.classifying_args)

