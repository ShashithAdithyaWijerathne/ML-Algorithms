import numpy as np


class Tree:
    def __init__(self, y, attr, cutoff):

        # class distribution
        self.probs = np.sum(y, axis=0) / len(y)

        # decision attribute & decision value
        self.attr = attr
        self.cutoff = cutoff

        # child nodes
        self.left = None
        self.right = None

    def predict(self, X, output_prob=False):

        # create prediction matrix
        m, n = len(X), len(self.probs)
        pred = np.zeros([m, n])

        # fill prediction matrix
        for idx, x in enumerate(X):

            # get probabilities
            probs = self._predict(x)

            if output_prob:
                probs = self.hot_encode(probs)

            # set distribution
            pred[idx, :] = probs

        return pred

    @staticmethod
    def hot_encode(probs):

        # pick all idx with highest value
        indices = np.where(probs == np.max(probs))

        # choose one
        idx = np.random.choice(*indices)

        # create hot-encoded vector
        hot = np.zeros(probs.shape)
        hot[idx] = 1

        return hot

    def _predict(self, x):

        # walk down the tree
        root = self
        while root:

            # return distribution if leaf node
            if not root.attr:
                return root.probs

            # check if attribute value is less or equal than cutoff
            is_left = x[root.attr] <= root.cutoff

            # go to left child, else right child
            root = root.left if is_left else root.right

        raise ValueError


class DecisionTree:

    def __init__(self, n_attrs=None):
        self.n_attrs = n_attrs
        self.tree = None

    @staticmethod
    def entropy(Y):

        if not Y.size:
            return 0

        # compute value distribution
        probs = Y.sum(axis=0) / Y.shape[0]
        probs = probs[probs != 0]

        # compute entropy
        h = - sum(probs * np.log2(probs))

        return h

    def info_gain(self, x, Y, cutoff):

        # compute entropy before
        before = self.entropy(Y)

        # divide into two subsets
        subset_l = Y[x <= cutoff, :]
        subset_r = Y[x > cutoff, :]

        # compute each weighted entropy
        h_l = len(subset_l) * self.entropy(subset_l)
        h_r = len(subset_r) * self.entropy(subset_r)

        # compute information gain
        ig = before - (h_l + h_r) / len(Y)

        return ig

    def _select_attr(self, X, Y, attrs):

        # compute information-gain for each (attr, cutoff) combination
        gains = {}
        for attr in attrs:
            for cutoff in X[:, attr]:

                # compute information gain
                gain = self.info_gain(X[:, attr], Y, cutoff)

                # store info-gain
                gains[(attr, cutoff)] = gain

        # pick attr & cutoff with highest info-gain
        attr, cutoff = max(gains, key=gains.get)

        # return best attribute to split
        return attr, cutoff, gains[(attr, cutoff)]

    def select_attr(self, X, Y, n_attrs):

        # check for single instance
        if len(X) <= 1:
            return None, None

        # check for pure subset
        if len(Y) in np.sum(Y, axis=0):
            return None, None

        # pick attributes to check
        attrs = np.random.choice(X.shape[1], n_attrs, replace=False)

        # pick best attribute to split
        attr, cutoff, gain = self._select_attr(X, Y, attrs)

        # check for relevant gain
        if np.around(gain, decimals=6) <= 0:
            return None, None

        # return best split
        return attr, cutoff

    @staticmethod
    def split(X, Y, attr, cutoff):

        # return empty if no attribute
        if not attr:
            return None, None, None, None

        # define boolean idx vector
        lines = X[:, attr] <= cutoff

        # return two subsets
        return X[lines], Y[lines], X[~lines], Y[~lines]

    def grow(self, X, Y, n_attrs):

        # pick attribute to split
        attr, cutoff = self.select_attr(X, Y, n_attrs)

        # split data into subsets
        X_l, Y_l, X_r, Y_r = self.split(X, Y, attr, cutoff)

        # grow root node
        node = Tree(Y, attr, cutoff)

        # grow child-nodes
        if attr:
            node.left = self.grow(X_l, Y_l, n_attrs)
            node.right = self.grow(X_r, Y_r, n_attrs)

        return node

    def fit(self, X, Y):

        # number of attributes to choose from when spliting
        n_attrs = self.n_attrs or X.shape[1]

        # fit decision tree
        self.tree = self.grow(X, Y, n_attrs)

    def predict(self, X, output_probs=False):
        return self.tree.predict(X, output_probs)
