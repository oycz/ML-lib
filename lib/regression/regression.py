class Regression:
    def __init__(self, data, training_method, training_args, classifying_method, classifying_args):
        self.training_method = training_method
        self.training_args = training_args
        self.classifying_method = classifying_method
        self.classifying_args = classifying_args
        self.data = data
        self.__train__()

    def __train__(self):
        training_result = self.training_method.train(self.data, self.training_args)
        self.model = training_result

    def predict(self, batch_data):
        return self.predict_method(batch_data, self.model, self.classifying_args)

