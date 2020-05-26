import numpy as np


def self(param):
    return param


def read_file(filename, splitter=",", line_splitter="\n", Y_index=-1, data_type=self):
    data = open(filename, "r+", encoding="UTF-8-sig")
    X, Y = [], []
    for line in data:
        arr_line = line.split(line_splitter)[0].split(splitter)
        X_unit = arr_line[1: Y_index]
        X_unit = [data_type(n) for n in X_unit]
        Y_unit = data_type(arr_line[Y_index])
        X += [X_unit]
        Y += [Y_unit]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
