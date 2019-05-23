from BPInterNode import *


class BPLeafNode(object):
    def __init__(self, m, parent=None, indices=[], data=[], next=None):
        self.m = m
        self.parent = parent
        self.indices = indices
        self.data = data
        self.next = next

    def insert(self, index, datum):
        """
        Add one record to a leaf node.
        """
        if len(self.indices) == 0:                  # new empty leaf, no need to check violation
            self.indices.append(index)
            self.data.append(datum)
            return
        ###############################################################
        # find the place to insert
        insert_point = 0
        min_index, max_index = self.indices[0], self.indices[-1]
        if index < min_index:
            self.indices = [index] + self.indices
            self.data = [datum] + self.data
        elif max_index < index:
            self.indices.append(index)
            self.data.append(datum)
        else:
            for i in range(len(self.indices) - 1):
                if self.indices[i] < index < self.indices[i + 1]:
                    insert_point = i
                    break
            self.indices.insert(insert_point + 1, index)
            self.data.insert(insert_point + 1, datum)
        #####################################################################
        #  check violation
        if len(self.indices) > self.m - 1:                  # split a leaf node
            median_index = len(self.indices) // 2
            median = self.indices[median_index]

            right_indices = self.indices[median_index:]     # create a new split leaf
            right_data = self.data[median_index:]
            right_leaf = BPLeafNode(m=self.m, parent=self.parent, indices=right_indices, data=right_data)
            index = right_indices[0]

            self.next = right_leaf                          # link the new leaf to the current leaf
            self.indices = self.indices[:median_index]      # split the data in the current leaf
            self.data = self.data[:median_index]

            if self.parent is None:                         # create an internal node as the new parent
                internal_node = BPInterNode(m=self.m, children=[self, right_leaf], indices=[median])

                right_leaf.parent = internal_node           # link children to the new parent
                self.parent = internal_node
                return internal_node

            else:
                return self.parent.insert(index, right_leaf)       # Upload the index and the leaf

