import numpy as np
from collections import Counter


# Decision tree with IC3 algorithm
class DecisionTree:

    def __init__(self, X, Y, feature_names, path_to_it='', parent=None):
        global number
        self.number = number
        number += 1
        self.X = X
        self.Y = Y
        self.feature_names = feature_names
        self.path_to_it = path_to_it
        self.IGs, self.max_IG_index, self.max_IG = None, None, None

        self.children = None
        self.X_items, self.Y_items, self.Y_label_count = self.get_item_labels()
        self.is_leaf, self.label = False, None
        self.index_to_split = -1
        self.process()

    def get_item_labels(self):
        X_items = [None] * self.X.shape[1]
        for j in range(self.X.shape[1]):
            row = self.X[:, j]
            X_items[j] = list(Counter(row).keys())
        Y_label_count = Counter(self.Y)
        Y_items = list(Y_label_count.keys())
        return X_items, Y_items, Y_label_count

    def calc_max_IG(self):
        IGs = np.zeros(self.X.shape[1])
        for j in range(self.X.shape[1]):
            IG = self.entropy(j)
            IGs[j] = IG
        self.IGs = IGs
        self.max_IG_index = np.argmin(IGs)
        self.max_IG = DecisionTree.total_entropy(self.Y) - self.entropy(self.max_IG_index)

        global all_IGs
        all_IGs += [np.sort(IGs)]

    def process(self):
        # end if
        if self.X.shape[0] == 0:
            # remais no dataset
            self.is_leaf = True
            self.label = None
        elif len(self.Y_label_count) == 1:
            # remains only one label
            self.is_leaf = True
            self.label = self.Y_label_count.most_common()[0][0]
        else:
            self.calc_max_IG()
            # else recurse
            # max information gain
            IGs = np.zeros(self.X.shape[1])
            for j in range(self.X.shape[1]):
                IG = self.entropy(j)
                IGs[j] = IG
            self.index_to_split = np.argmin(IGs)
            self.children = self.split(self.index_to_split)

    def split(self, index_to_split):
        children = dict()
        items = self.X_items[index_to_split]
        X = np.delete(self.X, index_to_split, axis=1)
        new_feature_names = np.delete(self.feature_names, index_to_split)
        for item in items:
            index = self.X[:, index_to_split] == item
            X_subset, Y_subset = X[index], self.Y[index]
            # 0: not / 1: yes
            children[item] = DecisionTree(X_subset, Y_subset, new_feature_names, parent=self, G=self.G, path_to_it=item)
        return children

    # x: single line
    def predict_single(self, x):
        if self.is_leaf:
            return self.label
        else:
            feature_to_predict = x[self.index_to_split]
            next_x = np.delete(x, self.index_to_split)
            if not self.children.__contains__(feature_to_predict):
                return None
            predict_result = self.children[feature_to_predict].predict_single(next_x)
            if predict_result:
                return predict_result
            else:
                return self.Y_label_count.most_common()[0][0]

    # conditional entropy of Y knowing row n
    def entropy(self, n):
        d = dict()
        for i in range(self.X.shape[0]):
            ch = self.X[i][n]
            l = self.Y[i]
            if not d.__contains__(ch):
                d[ch] = dict()
                d[ch]['n'] = 0
                d[ch]['labels'] = dict()
            d[ch]['n'] += 1
            if not d[ch]['labels'].__contains__(l):
                d[ch]['labels'][l] = 0
            d[ch]['labels'][l] += 1
        e = 0.0
        tot = self.X.shape[0]
        for feature in d:
            f_num = d[feature]['n']
            p = f_num / tot
            u = 0.0
            labels = d[feature]['labels']
            for label in labels:
                label_p = labels[label] / f_num
                u += label_p * np.log2(label_p)
            e -= p * u
        return e

    @staticmethod
    # entropy
    def total_entropy(Y):
        labels = Counter(Y)
        tot = Y.size
        e = 0.0
        for label in labels:
            p = labels[label] / tot
            e = e - p * np.log2(p)
        return e

    @staticmethod
    def read_file(filename):
        data = open(filename, "r+", encoding="UTF-8-sig")
        X, Y = [], []
        for line in data:
            arr_line = line.split("\n")[0].split(",")
            X_unit = arr_line[1:]
            Y_unit = arr_line[0]
            X += [X_unit]
            Y += [Y_unit]
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # predict a matrix
    @staticmethod
    def predict(dt, X):
        predicts = []
        for i in range(X.shape[0]):
            x = X[i]
            predicts += [dt.predict_single(x)]
        return predicts

    @staticmethod
    def calc_accu(t, test_X, test_Y):
        pred = DecisionTree.predict(t, test_X)
        accu = np.sum(test_Y == pred) / test_Y.shape[0]
        return accu


if __name__ == "__main__":
    feature_names = np.array(['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment',
                     'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                     'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                     'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                     'ring-type', 'spore-print-color', 'population', 'habitat'])

    train_X, train_Y = DecisionTree.read_file("train.data")
    test_X, test_Y = DecisionTree.read_file("test.data")

    t = DecisionTree(train_X, train_Y, feature_names)
    accu = DecisionTree.calc_accu(t, test_X, test_Y)
