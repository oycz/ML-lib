
def read_regression_data(splitter=','):
    data = open("perceptron.data")
    X, Y = [], []
    for line in data:
        arr_line = line.split("\n")[0].split(splitter)
        X_unit = arr_line[0:-1]
        Y_unit = arr_line[-1]
        X += [X_unit]
        Y += Y_unit
    return X, Y