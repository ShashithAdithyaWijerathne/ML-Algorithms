import numpy as np

from decision_tree import DecisionTree


class Forest:

    def __init__(self, trees, n_classes):
        self.trees = trees
        self.classes = n_classes

    def predict(self, X, output_probs=False):

        # get avg of all trees
        pred = np.zeros([len(X), self.classes])
        for tree in self.trees:
            pred += tree.predict(X, output_probs=True)
        pred /= len(self.trees)

        # return ditribution
        if output_probs:
            return pred

        # return hot encoded
        for idx, probs in enumerate(pred):

            # choose most probable column
            p_indices = np.where(probs == probs.max())
            p_idx = np.random.choice(*p_indices)
            pred[idx, :] = 0
            pred[idx, p_idx] = 1

        return pred


class RandomForest:

    def __init__(self, n_attrs=None):
        self.n_attrs = n_attrs
        self.forest = None

    @staticmethod
    def bootstrap(X, Y):
        # random pick line indices with replacement
        lines = np.random.choice(len(X), len(X), replace=True)
        return X[lines, :], Y[lines, :]

    def grow(self, X, Y, n_trees=100):

        # number of attributes to choose from when splitting
        n_attrs = self.n_attrs or int(np.sqrt(X.shape[1]))

        trees = []
        for _ in range(n_trees):

            # bootstrap sample
            X_boot, Y_boot = self.bootstrap(X, Y)

            # grow tree
            model = DecisionTree(n_attrs)
            model.fit(X_boot, Y_boot)

            # store tree
            trees.append(model)

        return Forest(trees, Y.shape[1])

    def fit(self, X, Y):
        self.forest = self.grow(X, Y)

    def predict(self, X, output_probs=False):
        return self.forest.predict(X, output_probs)
