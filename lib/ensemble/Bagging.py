import numpy as np
import pygraphviz as pg

node_num = 0


class DecisionTree:
    def __init__(self, attr_split_index=None, parent=None):
        self.label = None
        self.attr_split_index = attr_split_index
        self.tsub, self.fsub = None, None
        self.parent = parent

        global node_num
        self.name = str(node_num) + "." + str(self.attr_split_index)
        node_num = node_num + 1

    def pred(self, x):
        if self.label is not None:
            return self.label
        else:
            split_value = x[self.attr_split_index]
            if split_value == -1:
                return self.fsub.pred(x)
            else:
                return self.tsub.pred(x)

    def pred_all(self, X):
        preds = []
        for x in X:
            preds += [self.pred(x)]
        return np.array(preds)


def read_file(filename):
    data = open(filename, "r+", encoding="UTF-8-sig")
    X, Y = [], []
    for line in data:
        arr_line = line.split("\n")[0].split(",")
        X_unit = [1 if i == '1' else -1 for i in arr_line[1:]]
        Y_unit = 1 if arr_line[0] == '1' else -1
        X += [X_unit]
        Y += [Y_unit]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def bootstrap(X, Y):
    m = X.shape[0]
    sample_X, sample_Y = np.zeros(X.shape), np.zeros(Y.shape)
    for i in range(m):
        r = np.random.randint(m)
        sample_X[i] = X[r]
        sample_Y[i] = Y[r]
    return sample_X, sample_Y


def bagging_train(X, Y, ts, round=20):
    predictors = []
    for i in range(round):
        sample_X, sample_Y = bootstrap(X, Y)
        t = best_tree(ts, sample_X, sample_Y)
        predictors += [t]
    return predictors


def bagging_predict(X, predictors):
    predict = np.zeros(X.shape[0])
    for t in predictors:
        pred_Ys = t.pred_all(X)
        predict += pred_Ys
    # print(np.sign(predict))
    return np.sign(predict)


def single_layer_tree(attr_split_index, l1, l2):
    t = DecisionTree(attr_split_index)
    t.fsub = DecisionTree()
    t.tsub = DecisionTree()
    t.fsub.label, t.tsub.label = l1, l2
    return t


def best_tree(ts, X, Y):
    best_accu = float("-inf")
    best_k = -1
    accus = dict()
    for k in ts:
        t = ts[k]
        pred_Ys = t.pred_all(X)
        accu = np.sum(pred_Ys == Y) / Y.size
        accus[k] = [accu]
        if accu > best_accu:
            best_accu = accu
            best_k = k
    return ts[best_k]


def draw_tree(t, filename):
    G = pg.AGraph(strict=False, directed=True)
    draw_tree_helper(G, t)
    G.layout(prog='dot')
    G.draw(str(filename) + '.png')


def draw_tree_helper(G, t):
    G.add_node(t.name)
    n = G.get_node(t.name)
    if t.label is not None:
        if t.label == 1:
            n.attr['label'] = "1"
            n.attr['color'] = 'red'
        else:
            n.attr['label'] = "-1"
            n.attr['color'] = 'green'
    else:
        n.attr['label'] = t.attr_split_index
        # t subtree
        tsub = t.tsub
        G.add_edge(t.name, tsub.name)
        e = G.get_edge(t.name, tsub.name)
        e.attr['label'] = "t"
        draw_tree_helper(G, tsub)
        # f subtree
        fsub = t.fsub
        G.add_edge(t.name, fsub.name)
        e = G.get_edge(t.name, fsub.name)
        e.attr['label'] = "f"
        draw_tree_helper(G, fsub)


if __name__ == "__main__":
    heart_train_X, heart_train_Y = read_file("train.data")
    heart_test_X, heart_test_Y = read_file("test.data")

    m = heart_train_X.shape[0]
    n = heart_train_X.shape[1]

    ts = dict()
    for i in range(n):
        ts[i, -1, -1] = single_layer_tree(i, -1, -1)
        ts[i, -1, 1] = single_layer_tree(i, -1, 1)
        ts[i, 1, -1] = single_layer_tree(i, 1, -1)
        ts[i, 1, 1] = single_layer_tree(i, 1, 1)

    predictors = bagging_train(heart_train_X, heart_train_Y, ts, round=20)

    for i in range(len(predictors)):
        t = predictors[i]
        draw_tree(t, i)

    pred_train_Y = bagging_predict(heart_train_X, predictors)
    pred_test_Y = bagging_predict(heart_test_X, predictors)

    accu_train = np.sum(pred_train_Y == heart_train_Y) / heart_train_Y.size
    accu_test = np.sum(pred_test_Y == heart_test_Y) / heart_test_Y.size
    print(accu_train)
    print(accu_test)
