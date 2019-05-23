from BPTree import *


if __name__ == '__main__':
    tree = BPTree(5)

    # indices = [5, 8, 10, 15, 16]
    indices = [i for i in range(1, 40)]
    i = 0
    for index in indices:
        print(i, "-th insertion ... ...")
        tree.insert(index, str(index))
        tree.traversal(tree.root)
        i += 1
    print("traversal")
    print(tree.root.children[0].parent.indices)
    # tree.traversal(tree.root.children[0].children[0].parent.indices)