class Relation(object):
    """
    Describe a relation in database.

    Attributes:
        tuple_per_block: How many tuples fit a block.
        num_tuple: Number of tuples.
        num_block: Number of blocks to store the relation.
    """
    def __init__(self, rname, info, data=[], num_tuple=0):
        self.rname = rname
        self.info = info
        self.attributes = list(info.keys())
        self.byte_per_tuple = 0
        for v in info.values():
            self.byte_per_tuple += v
        self.num_tuple = num_tuple
        self.data = data
        self.primary_key = ''
        self.tuple_per_block = 0
        self.num_block = 0