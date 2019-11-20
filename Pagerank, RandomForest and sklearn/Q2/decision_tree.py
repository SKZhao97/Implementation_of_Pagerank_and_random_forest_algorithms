from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        #self.tree = {}
        pass
        self.tree = {}

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        pass

        def classLabel(labels):
            # Return the majority label (0 or 1) in labels list.
            oneLabel = sum(labels)
            zeroLabel = len(labels) - oneLabel
            if zeroLabel >= oneLabel:
                return 0
            return 1

        def treeBuilding(X, y):

            if sum(y) == len(y) or sum(y) == 0:
                return classLabel(y)

            tree = {}
            cols = len(X[0])
            #tree['entropy'] = entropy(y)
            #tree['num_rows'] = len(X)
            #tree['num_one'] = sum(y)
            #tree['num_zero'] = len(y)-sum(y)

            info_gain_list = []
            split_val_list = []

            for idx in range(cols):
                best_val = 0
                best_gain = 0
                col = [row[idx] for row in X]
                if isinstance(col[0], str):
                    val_set = set(col)
                    for val in val_set:
                        X_left, X_right, y_left, y_right = partition_classes(X, y, idx, val)
                        if len(y_left) == 0 or len(y_right) == 0:
                            break

                        gain = information_gain(y, [y_left, y_right])
                        if gain > best_gain:
                            best_gain = gain
                            best_val = val
                else:
                    steps = np.linspace(start=np.min(col), stop=np.max(col), num=5, endpoint=False)[1:]
                    for val in steps:
                        X_left, X_right, y_left, y_right = partition_classes(X, y, idx, val)
                        if len(y_left) == 0 or len(y_right) == 0:
                            break
                        gain = information_gain(y, [y_left, y_right])
                        if gain > best_gain:
                            best_gain = gain
                            best_val = val

                info_gain_list.append(best_gain)
                split_val_list.append(best_val)

            best_split_col = np.argmax(info_gain_list)
            best_split_value = split_val_list[best_split_col]
            X_left, X_right, y_left, y_right = partition_classes(X, y, best_split_col, best_split_value)
            tree['split_attribute'] = best_split_col
            tree['split_value'] = best_split_value
            tree['left_child'] = treeBuilding(X_left, y_left)
            tree['right_child'] = treeBuilding(X_right, y_right)
            return tree

        self.tree = treeBuilding(X, y)


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        pass

        def recurClassify(tree, case):
            if tree == 0 or tree == 1:
                return tree
            else:
                attr = tree['split_attribute']
                val = tree['split_value']

                if isinstance(val, str):
                    if record[attr] == val:

                        return recurClassify(tree['left_child'], case)
                    else:
                        return recurClassify(tree['right_child'], case)

                else:
                    if record[attr] <= val:
                        return recurClassify(tree['left_child'], case)

                    else:
                        return recurClassify(tree['right_child'], case)
        return recurClassify(self.tree, record)

