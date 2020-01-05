class Regression:

    def __init__(self, X, Y, training_method, classifying_method, training_args=[], classifying_args=[]):
        self.training_method = training_method
        self.training_args = training_args
        self.classifying_method = classifying_method
        self.classifying_args = classifying_args
        self.X = X
        self.Y = Y
        self.__train__()

    def __train__(self):
        training_result = self.training_method(self.X, self.Y)
        self.model = training_result

    def predict(self, X):
        return self.predict_method(X, self.model, self.classifying_args)

