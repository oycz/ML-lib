from collections import Counter
import numpy as np
import pygraphviz as pg
from lib.preprocessing.read_data import read_file


node_num = 0


class DecisionTree:
    def __init__(self, X, Y, attr_vector, parent=None):
        self.X = X
        self.Y = Y
        self.attr_vector = attr_vector
        self.counter = Counter(self.Y).most_common()
        self.label = None
        if len(self.counter) > 0:
            self.label = self.counter[0][0]
        self.is_node = True
        self.attr_split_index = -1
        self.tsub, self.fsub = None, None
        self.parent = parent

        global node_num
        self.name = str(node_num) + "." + str(self.attr_split_index)
        node_num = node_num + 1

    def pred(self, x):
        if self.is_node:
            return self.label
        else:
            split_value = x[self.attr_split_index]
            if split_value == -1:
                val = self.fsub.pred(x)
            else:
                val = self.tsub.pred(x)
            if val:
                return val
            else:
                return self.label

    def pred_all(self, X):
        preds = []
        for x in X:
            preds += [self.pred(x)]
        return np.array(preds)

    def split(self, index_to_split):
        if self.attr_split_index != -1:
            return
        else:
            self.attr_split_index = index_to_split
            new_attr_vector = np.array(self.attr_vector)
            new_attr_vector[index_to_split] = 0
            index_t = self.X[:, index_to_split] == 1
            index_f = self.X[:, index_to_split] == -1
            X_subset_t, Y_subset_t = self.X[index_t], self.Y[index_t]
            X_subset_f, Y_subset_f = self.X[index_f], self.Y[index_f]
            self.tsub = DecisionTree(X_subset_t, Y_subset_t, new_attr_vector, parent=self)
            self.fsub = DecisionTree(X_subset_f, Y_subset_f, new_attr_vector, parent=self)
            self.is_node = False


def two_split_tree(X, Y, one, two, split_on_true):
    attr_vector = np.ones(X.shape[1])
    t = DecisionTree(X, Y, attr_vector)
    t.split(one)
    if split_on_true:
        t.tsub.split(two)
    else:
        t.fsub.split(two)
    return t


def best_tree(ts, X, Y, weight):
    min_sum_weight, min_key = float("inf"), -1
    sum_weights = dict()
    pred_Ys = dict()
    for k in ts:
        t = ts[k]
        pred_Y = t.pred_all(X)
        sum_weight = np.sum(((pred_Y != Y) + 0) * weight)
        sum_weights[k] = sum_weight
        pred_Ys[k] = pred_Y
        if sum_weight < min_sum_weight:
            min_sum_weight = sum_weight
            min_key = k
    return ts[min_key], sum_weights[min_key], pred_Ys[min_key]


def adaboost_pred(alphas, predictors, X, iter_num=None):
    if iter_num is None:
        iter_num = len(alphas)
    weighted_Y = np.zeros(X.shape[0])
    for i in range(iter_num):
        alpha = alphas[i]
        predictor = predictors[i]
        Y = predictor.pred_all(X)
        weighted_Y += alpha * Y
    print(iter_num)
    print(weighted_Y)
    return np.sign(weighted_Y)


def adaboost(X, Y, boost_n):
    m, n = X.shape[0], X.shape[1]
    weight = np.full(m, 1 / m)
    weights = []
    epsilons = []
    alphas = []
    predictors = []

    for i in range(boost_n):
        ts = dict()
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                ts[a, b, 1] = two_split_tree(X, Y, a, b, True)
                ts[a, b, 0] = two_split_tree(X, Y, a, b, False)
        t, epsilon, pred_Y = best_tree(ts, X, Y, weight)

        # calculate alpha
        alpha = (1/2) * np.log((1-epsilon)/epsilon)

        # one boost result
        predictors += [t]
        weights += [weight]
        epsilons += [epsilon]
        alphas += [alpha]

        # update weight
        weight = (weight * np.exp(-Y*pred_Y*alpha)) / (2 * np.sqrt(epsilon * (1-epsilon)))

    return epsilons, alphas, predictors

def draw_tree(t, filename):
    G = pg.AGraph(strict=False, directed=True)
    draw_tree_helper(G, t)
    G.layout(prog='dot')
    G.draw(filename + '.png')


def draw_tree_helper(G, t):
    G.add_node(t.name)
    n = G.get_node(t.name)
    if t.is_node:
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

    # 1 (a)
    epsilons1, alphas1, predictors1 = adaboost(heart_train_X, heart_train_Y, 5)

    ada_pred_Y = adaboost_pred(alphas1, predictors1, heart_train_X)
    accu = np.sum(ada_pred_Y == heart_train_Y) / heart_train_Y.size

    # draw tree
    for i in range(len(predictors1)):
        t = predictors1[i]
        draw_tree(t, str(i))

    # 1 (b)
    epsilons2, alphas2, predictors2 = adaboost(heart_train_X, heart_train_Y, 10)
    q2_accu_train = []
    q2_accu_test = []
    # draw tree
    for i in range(len(predictors2)):
        t = predictors2[i]
        draw_tree(t, str(i))
    for i in range(1, 11):
        train_pred = adaboost_pred(alphas2, predictors2, heart_train_X, iter_num=i)
        train_accu = np.sum(train_pred == heart_train_Y) / heart_train_Y.size
        q2_accu_train += [train_accu]

        test_pred = adaboost_pred(alphas2, predictors2, heart_test_X, iter_num=i)
        test_accu = np.sum(test_pred == heart_test_Y) / heart_test_Y.size
        q2_accu_test += [test_accu]
