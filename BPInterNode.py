class BPInterNode(object):
    """
    An internal node of a b+ tree object.

    Attributes:
        children: An internal node can hold at most m children.
    """

    def __init__(self, m, children=None, indices=[]):
        self.parent = None
        self.children = children
        self.indices = indices
        self.m = m

    def add_index(self, index):
        self.indices.append(index)

    def insert(self, index, leaf):
        """
        Find a place to insert the children
        :param leaf:
        :return:
        """
        ###############################################################
        # find the place to insert
        # median = leaf.indices[0]
        min_index, max_index = self.indices[0], self.indices[-1]
        leaf.parent = self
        if index < min_index:
            self.indices = [index] + self.indices
            self.children.insert(1, leaf)
        elif max_index < index:
            self.indices.append(index)  # link the children to parent
            self.children.append(leaf)
        else:
            insert_point = 0
            for i in range(len(self.indices) - 1):
                if self.indices[i] < index < self.indices[i + 1]:
                    insert_point = i
                    break
            self.indices.insert(insert_point + 1, index)
            self.children.insert(insert_point + 2, leaf)

        if len(self.indices) > self.m - 1:                  # trigger violation
            median_index = len(self.indices) // 2
            median = self.indices[median_index]

            right_indices = self.indices[median_index + 1:]
            right_children = self.children[median_index + 1:]
            right_node = BPInterNode(children=right_children, indices=right_indices, m=self.m)
            for child in right_children:
                child.parent = right_node

            self.indices = self.indices[:median_index]      # split an internal node
            self.children = self.children[:median_index + 1]

            if self.parent is None:
                internal_node = BPInterNode(m=self.m, children=[self, right_node], indices=[median])
                right_node.parent = internal_node
                self.parent = internal_node
                return internal_node
            else:
                return self.parent.insert(median, right_node)

    def insert_to_inner(self, node):
        pass


