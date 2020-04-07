from collections import deque
import math


class Tree:
    def __init__(self, series, attr):

        # target attribute distribution & name
        probs = series.value_counts(normalize=True)
        self.probs = probs.to_dict()
        self.target = series.name

        # decision attribute name & empty child nodes
        self.attr = attr if attr else 'pure'
        self.nodes = {}

    def inference(self, values):

        root = self
        while root:
            if not root.nodes:
                return root.probs
            root = root.nodes[values[root.attr]]

    def print_bfs(self):

        queue = deque([self])
        while queue:

            # pick node
            root = queue.popleft()
            print(root.attr, '\n', root.target, root.probs)

            # enqueue child-nodes
            for node in root.nodes.values():
                queue.append(node)


class DecisionTreeCategorical:

    def __init__(self):
        self.tree = None

    @staticmethod
    def entropy(series):

        # compute unique values probabilities
        probs = series.value_counts(normalize=True)

        # compute entropy
        h = - sum([p * math.log(p, 2) for p in probs])

        return h

    def info_gain(self, df, target, attr):

        # compute entropy before
        before = self.entropy(df[target])

        # pick subset for each attribute value
        series = df[attr]
        subsets = (df[series == value] for value in series.unique())

        # compute each weighted subset entropy given target variable
        entropies = (len(subset) * self.entropy(subset[target]) for subset in subsets)

        # compute information gain
        ig = before - sum(entropies) / len(df)

        return ig

    def info_gain_ratio(self, df, target, attr):

        # compute information-gain & intrinsic value
        ig = self.info_gain(df, target, attr)
        iv = self.entropy(df[attr])

        # compute ratio
        ratio = ig / iv

        return ratio

    def split_on(self, df, target):

        # no split if pure subset
        domain = df[target].unique()
        if len(domain) == 1:
            return ''

        # list attributes to check
        attrs = (attr for attr in df if attr != target)

        # find atrribute with hightest information-gain
        best = max(attrs, key=lambda attr: self.info_gain_ratio(df, target, attr))

        return best

    @staticmethod
    def splits(df, attr):

        # return if no attribute
        if not attr:
            return {}

        # pick data subset with each attribute value
        subsets = {}
        series = df[attr]
        for value in series.unique():

            # pick subset
            subset = df[series == value]

            # drop attribute
            subset = subset.drop(attr, axis=1)

            # store subset
            subsets[value] = subset

        return subsets

    def grow(self, df, target):

        # pick attribute to split
        attr = self.split_on(df, target)

        # split data into subsets
        subsets = self.splits(df, attr)

        # create tree-node
        root = Tree(df[target], attr)

        # add child-nodes
        root.nodes = {value: self.grow(subset, target) for value, subset in subsets.items()}

        return root

    def fit(self, df, target):
        self.tree = self.grow(df, target)

    def predict_one(self, values):
        return self.tree.inference(values)
