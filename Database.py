import random
from Buffer import *
from Relation import *
from BPTree import *
import pysnooper


class DBMS(object):
    """
    A buffer holds 8 blocks, each block holds 8 tuples, each tuple takes 8 bytes.
    """
    def __init__(self):
        self.relations = {}
        self.selection_algorithms = ['scanning', 'binary', 'indexed']
        self.buf_size = 512
        self.blk_size = 64
        self.buffer = Buffer(self.buf_size, self.blk_size)
        self.tree = ''
        self.routing_table_join = {'nested_loop_join', 'sort_merge_join', 'hash_join'}

    def make_relation(self, rname, num_tuple, range_A, range_B, v):
        """ Randomly produce the relation relation """
        data = []
        integers = []
        for i in range(num_tuple):
            value_A, value_B = random.randint(range_A[0], range_A[1]), random.randint(range_B[0], range_B[1])
            integers += [value_A, value_B]
            data += [value_A] + [value_B]
            if value_A == v:
                print(value_A, value_B)
        relation = Relation(rname=rname, info={'A': 4, 'B': 4}, data=data, num_tuple=num_tuple)

        ts = []
        for i in range(0, len(integers) - 1, 2):
            ts.append([integers[i], integers[i + 1]])

        def take_first(elem):
            return elem[0]

        ts.sort(key=take_first)
        f = open('./tsdata_' + rname.lower() + '.txt', 'w')
        for tsdata in ts:
            value_1 = tsdata[0]
            value_2 = tsdata[1]
            f.write(str(value_1) + " " + str(value_2) + "\n")
        print(rname, ts)
        # save the record in the buffer
        # num_block: The number of num_block needed to store the relation
        # tuple_per_block: The number of tuples in one block
        # write a block to the hard disk in each iteration
        num_block = num_tuple * relation.byte_per_tuple // self.blk_size
        tuple_per_block = self.blk_size // relation.byte_per_tuple
        relation.tuple_per_block = tuple_per_block
        relation.num_block = num_block
        for i in range(num_block):
            attribute_per_block = tuple_per_block * len(relation.attributes)
            d_start = i * attribute_per_block   # fetch a block of data
            d_end = d_start + attribute_per_block - 1
            blk_data = data[d_start:d_end + 1]

            if self.buffer.is_full():           # apply for a block to store
                self.buffer.free_buffer()
            blk = self.buffer.get_new_block()

            start = blk * self.blk_size * 8     # write data to the block integer-wise
            for datum in blk_data:
                datum = '{0:b}'.format(datum)
                zeros = '0' * (32 - len(datum))
                datum = zeros + datum
                for j in range(32):             # write an integer bit-by-bit
                    self.buffer.data[start + j] = datum[j]
                start += 32

            addr = 'relation_' + relation.rname.lower() + '_{index}'.format(index=i)  # write the data to the disk
            self.buffer.write_blk_to_disk(blk, addr)

        self.relations[rname] = relation

    def select(self, relation='R', value=40, algorithm='scanning'):
        """ Execute the selection based on the algorithm given. """
        if algorithm not in self.selection_algorithms:
            print("No such selection algorithm")
            return
        if algorithm == 'scanning':
            return self.select_scanned_based(relation, value)
        elif algorithm == 'binary':
            return self.select_binary_search(relation, value)
        elif algorithm == 'indexed':
            return self.select_indexed_based(relation, value)

    def select_scanned_based(self, rname, value):
        """
        Execute the selection operation based on scanning.
        Read one block of R into one block in the buffer,
        then select one tuple at a time in the block,
        finally export the selected tuples to another file.
        """
        print("Searching for", value, ", based on linear scanning")
        selected, results = [], []
        old_IO = self.buffer.num_IO
        for i in range(self.relations[rname].num_block):
            file = 'relation_' + rname.lower() + '_' + str(i)
            self.buffer.read_blk_from_disk(file)
            data = self.buffer.data[0: self.blk_size * 8]

            # In the inner loop, process one tuple at a time
            # start address of each tuple
            bit_per_tuple = self.relations[rname].byte_per_tuple * 8
            t_start, t_end = 0, bit_per_tuple - 1
            for j in range(self.relations[rname].tuple_per_block):
                tuple = data[t_start:t_end + 1]
                value_A = int("".join('%s' % i for i in tuple[0:32]), 2)
                value_B = int("".join('%s' % i for i in tuple[32:64]), 2)
                if value_A == value:
                    selected += tuple
                    results.append([value_A, value_B])
                t_start += bit_per_tuple
                t_end += bit_per_tuple

            self.buffer.free_blk(0)

        if not results:
            return False
        # Write the selected tuples to file
        file = rname.lower() + '_select_' + str(value) + '.txt'
        dir = os.path.dirname(__file__)
        path = dir + '/blk_data/' + file
        with open(path, 'w') as f:
            for bit in selected:
                f.write(str(bit))
        print("IO times", self.buffer.num_IO - old_IO)
        return 'Query results', results

    def select_indexed_based(self, rname, value):
        """
        Step 1: Scan all the files and build a B+ tree.
        Step 2: Search
        """
        print("Searching for", value, ", using indexing technology")
        self.build_index(rname)
        old_IO = self.buffer.num_IO
        leaf = self.tree.search(self.tree.root, value)
        indices = leaf.indices
        data = leaf.data

        if value not in indices:
            return False
        position = indices.index(value)
        links = data[position]
        self.buffer.free_buffer()
        links = set(links)
        results = []
        for link in links:
            block = self.buffer.read_blk_from_disk(link)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for i in range(0, len(integers), 2):
                tuples.append([integers[i], integers[i + 1]])
            for t in tuples:
                if t[0] == value:
                    results.append(t)
        print("IO times", self.buffer.num_IO - old_IO)
        return 'Query results',results

    def select_binary_search(self, rname, value):
        """
        First perform external merge sort,
        then search block by block.
        :param rname: The name of the relation.
        :param value: A value as the join condition.
        :return:
        """
        print("Searching for", value, ", using external merge sort and binary search")
        N = self.relations[rname].num_tuple
        B = self.relations[rname].tuple_per_block
        M = self.buffer.num_all_blk
        if N <= B * (M**2):
            self.two_pass_multiway_external_merge_sort(rname)
        else:
            self.multi_pass_multiway_external_merge_sort(rname)
        old_IO = self.buffer.num_IO
        # Linear searching
        left, right, start = 0, N // B, 0
        found, terminate = False, False
        forward, backward = False, False
        # Find the file where the value first appear
        while left <= right:
            mid = int(left + (right - left)/2)
            filename = 'relation_' + rname.lower() + '_{index}'.format(index=mid)
            block = self.buffer.read_blk_from_disk(filename)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for i in range(0, len(integers) - 1, 2):
                tuples.append([integers[i], integers[i + 1]])
            t_min, t_max = tuples[0][0], tuples[-1][0]

            for t in tuples:
                if t[0] == value and not found:
                    found = True
            if found:
                break
            if value < t_min:
                right = mid
            elif t_max < value:
                left = mid
            else:
                print("No such record!")
                return False

        # Search results backward
        results = []
        start = mid
        while not terminate and start in range(N // B):
            filename = 'relation_' + rname.lower() + '_{index}'.format(index=start)
            block = self.buffer.read_blk_from_disk(filename)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for i in range(0, len(integers) - 1, 2):
                tuples.append([integers[i], integers[i + 1]])
            t_min, t_max = tuples[0][0], tuples[-1][0]
            backward = (t_max == value)
            forward = (t_min == value)
            for t in tuples:
                if t[0] == value:
                    results.append(t)
            if backward:
                start += 1
            elif forward:
                start = mid - 1
                mid -= 1
            else:
                terminate = True
        print("IO times", self.buffer.num_IO - old_IO)
        return 'Query results', results

    def two_pass_multiway_external_merge_sort(self, rname):
        # Pass 0: Read in M blocks at a time to produce sorted outcome.
        # Fill up the buffer blocks
        N = self.relations[rname].num_tuple
        B = self.relations[rname].tuple_per_block
        M = self.buffer.num_all_blk
        num_file = self.relations[rname].num_block
        for i in range(num_file):
            file = 'relation_' + rname.lower() + '_' + str(i)
            if not self.buffer.is_full():
                self.buffer.read_blk_from_disk(file)
            if self.buffer.is_full():
                data = self.buffer.data
                byte_per_tuple = self.relations[rname].byte_per_tuple
                num_tuple = len(data) // (byte_per_tuple * 8)
                # Start inner sorting
                for t1 in range(num_tuple):
                    for t2 in range(t1 + 1, num_tuple):
                        t1_start = t1 * byte_per_tuple * 8
                        t1_end = t1_start + byte_per_tuple * 8
                        t2_start = t2 * byte_per_tuple * 8
                        t2_end = t2_start + byte_per_tuple * 8
                        tuple1 = data[t1_start: t1_end]
                        tuple2 = data[t2_start: t2_end]

                        value1_1 = int("".join('%s' % i for i in tuple1[0:32]), 2)
                        value2_1 = int("".join('%s' % i for i in tuple2[0:32]), 2)

                        if value1_1 > value2_1:
                            # swap two tuples
                            temp = tuple1.copy()
                            data[t1_start: t1_end] = data[t2_start: t2_end]
                            data[t2_start: t2_end] = temp

                # Write the sorted result into intermediate file
                temp_file = 'temp_' + str(i % (byte_per_tuple - 1))
                self.buffer.write_buf_to_disk(temp_file)

        # Check whether the results are correct
        temp_file = 'temp_' + str(0)
        self.buffer.read_buf_from_disk(temp_file)
        integers = self.buffer.read_buf_int()
        tuples = []
        f = open('./blk_data/sort1.txt', 'w')
        for j in range(0, len(integers) - 1, 2):
            tuple = [integers[j], integers[j + 1]]
            tuples.append(tuple)
            f.write(str(integers[j]) + " " + str(integers[j + 1]) + "\n")
        self.buffer.free_buffer()

        #############################################################################
        # Pass 1: External merge sort
        #
        # Step 0: Initializing
        #
        # blk_pointer: Record how many blocks have been read for each file;
        # range 0~7 (M: how many blocks in a buffer)
        # tuple_pointer: Record how many tuples have been consumed for each block;
        # range 0~7 (B: tuple per block)

        num_sorted_file = N // (B * M)
        blk_pointer = [0] * num_sorted_file
        tuple_pointer = [0] * M

        # Step 1: Read in one block for every small files
        for i in range(num_sorted_file):
            temp_file = 'temp_' + str(i)
            self.buffer.read_blk_from_disk(temp_file, self.blk_size * 8 * blk_pointer[i])

        # Step 2: Merge-sort
        write_blk = self.buffer.get_new_block()
        write_pointer = write_blk * self.blk_size * 8
        count = 0
        while True:
            # Select a minimum from the blocks
            min_block, min_value, min_tuple = 0, 100000, []

            for block in range(num_sorted_file):  # Each block contributes its min
                if blk_pointer[block] == 8:       # Ignore what has been ran out
                    continue
                tuple_return, min_return = self.buffer.find_min(block, 8, tuple_pointer[block])
                if min_return < min_value:
                    min_value = min_return
                    min_block = block
                    min_tuple = tuple_return

            if min_value == 100000:
                break
            tuple_pointer[min_block] += 1

            # Write the selected tuple to the output block
            self.buffer.data[write_pointer: write_pointer + 64] = min_tuple.copy()
            write_pointer += 64

            # Write the ouptut buffer to disk when the output block is full
            if write_pointer >= (write_blk + 1) * self.blk_size * 8:
                addr = 'relation_' + rname.lower() + '_{index}'.format(index=count)
                count += 1
                self.buffer.write_blk_to_disk(write_blk, addr)

                # Re-apply for write block and reset the pointer
                write_blk = self.buffer.get_new_block()
                write_pointer = write_blk * self.blk_size * 8

            # Ignore it when the whole file has been processed
            # Otherwise, import next block of the file when the current block has been ran out
            if tuple_pointer[min_block] > B - 1:
                blk_pointer[min_block] += 1
                if blk_pointer[min_block] > M - 1:
                    continue
                temp_file = 'temp_' + str(min_block)
                self.buffer.free_blk(min_block)
                self.buffer.read_blk_from_disk(temp_file, self.blk_size * 8 * blk_pointer[min_block])
                tuple_pointer[min_block] = 0

        self.buffer.free_buffer()

    def multi_pass_multiway_external_merge_sort(self, rname):
        N = self.relations[rname].num_tuple
        B = self.relations[rname].tuple_per_block
        M = self.buffer.num_all_blk
        print("Preparing this function...")

    def build_index(self, rname):
        print("Build B+ tree index")
        self.tree = BPTree(5)
        N = self.relations[rname].num_block
        for i in range(N):
            file = 'relation_' + rname.lower() + '_' + str(i)
            if self.buffer.is_full():
                self.buffer.free_buffer()
            block = self.buffer.read_blk_from_disk(file)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for j in range(0, len(integers) - 1, 2):
                tuples.append([integers[j], integers[j + 1]])
            for t in tuples:
                key = t[0]
                value = file
                self.tree.insert(key, [value])

    def do_projection(self, rname):
        print("Projecting attribute...")
        self.buffer.free_buffer()
        results = []
        num_file = self.relations[rname].num_block
        write_block = self.buffer.get_new_block()
        w_start = write_block * self.blk_size * 8
        w_count = 0
        for i in range(num_file):
            addr = 'relation_' + rname.lower() + '_{index}'.format(index=i)
            if self.buffer.is_full():
                self.buffer.free_buffer()
            block = self.buffer.read_blk_from_disk(addr)
            t_start = block * self.blk_size * 8
            t_end = t_start + 64

            while t_start < (block + 1) * self.blk_size * 8:
                tuple = self.buffer.data[t_start:t_end]

                # Write to the output buffer
                self.buffer.data[w_start:w_start + 32] = tuple[0:32]
                value = int("".join('%s' % p for p in tuple[0:32]), 2)
                results.append(value)
                w_start += 32
                if w_start >= (write_block + 1) * self.blk_size * 8:
                    w_addr = 'project_' + rname.lower() + '_{index}'.format(index=w_count)
                    w_count += 1
                    self.buffer.write_blk_to_disk(write_block, w_addr)
                    write_block = self.buffer.get_new_block()
                    w_start = write_block * self.blk_size * 8
                t_start += 64
                t_end += 64

            self.buffer.free_blk(block)

        return 'Query results', results

    def nested_loop_join(self, r1, r2):
        print("Performing nested-loop join")
        relations = self.relations.keys()
        if r1 not in relations and r2 not in relations:
            return "The parameter relations do not exist!"
        elif r1 not in relations or r2 not in relations:
            return "One of the parameter relation does not exist!"
        self.buffer.free_buffer()

        # Block 0 is for writing
        # Block 1~6 holds blocks from relation 1
        # Block 7 holds one block from relation 2
        n1 = self.relations[r1].num_block
        n2 = self.relations[r2].num_block
        B = self.buffer.num_all_blk
        w_count = 0
        pointer = 0
        results = []
        old_IO = self.buffer.num_IO
        w_start = 0
        while pointer < n1:
            write_block = self.buffer.get_new_block()
            w_start = 0
            # First read M - 2 blocks of relation 1 into the buffer
            m = min(B - 2, n1 - pointer)
            for i in range(m):
                addr1 = 'relation_' + r1.lower() + '_{index}'.format(index=pointer)
                self.buffer.read_blk_from_disk(addr1)
                pointer += 1
            for i in range(B - m - 2):
                self.buffer.get_new_block()

            # Then read 1 block of relation 2 into the buffer
            for j in range(n2):
                addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=j)
                self.buffer.read_blk_from_disk(addr2)

                # Compare each tuple
                t1_start = self.blk_size * 8
                while t1_start < (m + 1) * self.blk_size * 8:
                    tuple_1 = self.buffer.data[t1_start: t1_start + 64]
                    value1_1 = int("".join('%s' % p for p in tuple_1[0:32]), 2)
                    value1_2 = int("".join('%s' % p for p in tuple_1[32:64]), 2)

                    # Read the block of the second relation
                    t2_start = (B - 1) * self.blk_size * 8
                    while t2_start < B * self.blk_size * 8:
                        tuple_2 = self.buffer.data[t2_start: t2_start + 64]
                        value2_1 = int("".join('%s' % p for p in tuple_2[0:32]), 2)
                        value2_2 = int("".join('%s' % p for p in tuple_2[32:64]), 2)
                        if value1_1 == value2_1:
                            results.append([value1_1, value1_2, value2_2])

                            # Write the tuples in the write buffer
                            t = tuple_1 + tuple_2[32:64]
                            if w_start + 96 > 512:                          # Time to write
                                rest_bit = 512 - w_start
                                self.buffer.data[w_start:512] = t[0:rest_bit]
                                w_addr = 'nest_join_' + r1.lower() + '_' + r2.lower() + '_{index}'.format(index=w_count)
                                self.buffer.write_blk_to_disk(0, w_addr)
                                w_count += 1
                                self.buffer.get_new_block()
                                self.buffer.data[0:96 - rest_bit] = t[rest_bit:]
                                w_start = 96 - rest_bit
                            else:
                                self.buffer.data[w_start:w_start + 96] = t
                                w_start += 96
                        # End if
                        t2_start += 64
                    # End while
                    t1_start += 64
                # End while
                self.buffer.free_blk(7)
            # End for
            self.buffer.free_buffer()
        # End while

        # Write the remaining bits in the write buffer
        if w_start > 0:
            w_addr = 'join_' + r1.lower() + '_' + r2.lower() + '_{index}'.format(index=w_count)
            self.buffer.write_blk_to_disk(0, w_addr)
        print("IO times", self.buffer.num_IO - old_IO)
        results.sort()
        return results

    def sort_merge_join(self, r1, r2):
        print("Performing sort-merge join")
        self.two_pass_multiway_external_merge_sort(r1)
        self.two_pass_multiway_external_merge_sort(r2)
        self.buffer.free_buffer()
        old_IO = self.buffer.num_IO
        B = self.buffer.num_all_blk
        n1 = self.relations[r1].num_block
        n2 = self.relations[r2].num_block
        self.buffer.get_new_block()
        w_count, w_start, pointer = 0, 0, 0
        prev_a, prev_joined, joined_file = -1, False, ''
        results = []
        for i in range(n1):
            addr1 = 'relation_' + r1.lower() + '_{index}'.format(index=i)
            self.buffer.read_blk_from_disk(addr1)
            t1_start = self.blk_size * 8

            if pointer >= 32:
                break
            addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=pointer)
            self.buffer.read_blk_from_disk(addr2)
            t2_start = 2 * self.blk_size * 8

            while t1_start < 2 * self.blk_size * 8 and pointer < 32:
                tuple1 = self.buffer.data[t1_start:t1_start + 64]
                a = int("".join('%s' % p for p in tuple1[0:32]), 2)
                b = int("".join('%s' % p for p in tuple1[32:64]), 2)

                # Load next block of r2, when block 2 is ran out
                if t2_start > 3 * self.blk_size * 8:
                    self.buffer.free_blk(2)
                    addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=pointer)
                    pointer += 1
                    if pointer >= 32:
                        break
                    self.buffer.read_blk_from_disk(addr2)
                    t2_start = 2 * self.blk_size * 8
                tuple2 = self.buffer.data[t2_start:t2_start + 64]
                c = int("".join('%s' % p for p in tuple2[0:32]), 2)
                d = int("".join('%s' % p for p in tuple2[32:64]), 2)
                t2_start += 64

                # First decide whether the current a euqals the previous a;
                # If yes, and the previous a did join operation,
                # then the current a should join the those tuples.
                if a == prev_a and prev_joined:
                    self.buffer.free_blk(2)
                    self.buffer.read_blk_from_disk(joined_file)
                    prev_t2_start = 2 * self.blk_size * 8
                    tuple2_ = self.buffer.data[prev_t2_start:prev_t2_start + 64]
                    c_ = int("".join('%s' % p for p in tuple2_[0:32]), 2)
                    d_ = int("".join('%s' % p for p in tuple2_[32:64]), 2)
                    prev_t2_start += 64

                    while a != c_:          # find the first matched tuple
                        tuple2_ = self.buffer.data[prev_t2_start:prev_t2_start + 64]
                        c_ = int("".join('%s' % p for p in tuple2_[0:32]), 2)
                        d_ = int("".join('%s' % p for p in tuple2_[32:64]), 2)
                        prev_t2_start += 64
                    if a == c_:
                        count = 0
                        while a == c_:
                            print([a, b, d_])
                            results.append([a, b, d_])
                            if prev_t2_start > 3 * self.blk_size * 8:
                                self.buffer.free_blk(2)
                                addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=joined_file)
                                if joined_file == '':
                                    joined_file = addr2
                                self.buffer.read_blk_from_disk(addr2)
                                count += 1
                                prev_t2_start = 2 * self.blk_size * 8

                            tuple2 = self.buffer.data[prev_t2_start:prev_t2_start + 64]
                            c_ = int("".join('%s' % p for p in tuple2[0:32]), 2)
                            d_ = int("".join('%s' % p for p in tuple2[32:64]), 2)
                            prev_t2_start += 64

                if a < c:
                    prev_joined = False
                    joined_file = ''
                    continue
                elif a > c:
                    count = 0
                    # Take next c until a == c or a < c
                    while a > c:
                        # Find next file if the current file is not sufficient
                        if t2_start > 3 * self.blk_size * 8:
                            self.buffer.free_blk(2)
                            addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=pointer + count)
                            self.buffer.read_blk_from_disk(addr2)
                            t2_start = 2 * self.blk_size * 8
                            count += 1

                        tuple2 = self.buffer.data[t2_start:t2_start + 64]
                        c = int("".join('%s' % p for p in tuple2[0:32]), 2)
                        t2_start += 64

                if a == c:
                    count = 0
                    while a == c:
                        print([a, b, d])
                        results.append([a, b, d])
                        prev_joined = True
                        if t2_start > 3 * self.blk_size * 8:
                            self.buffer.free_blk(2)
                            addr2 = 'relation_' + r2.lower() + '_{index}'.format(index=pointer + count)
                            if joined_file == '':
                                joined_file = addr2
                            self.buffer.read_blk_from_disk(addr2)
                            count += 1
                            t2_start = 2 * self.blk_size * 8

                        tuple2 = self.buffer.data[t2_start:t2_start + 64]
                        c = int("".join('%s' % p for p in tuple2[0:32]), 2)
                        d = int("".join('%s' % p for p in tuple2[32:64]), 2)
                        t2_start += 64
                else:
                    prev_joined = False
                    joined_file = ''
                t1_start += 64
                prev_a = a
            # Finish handling block 1 which holds data from relation 1
            self.buffer.free_blk(1)
        print("IO times", self.buffer.num_IO - old_IO)
        return results

    def hash_join(self, r1, r2):
        hash_map_1, hash_map_2 = {}, {}
        n1 = self.relations[r1].num_block
        n2 = self.relations[r2].num_block
        for i in range(n1):
            file = 'relation_' + r1.lower() + '_' + str(i)
            if self.buffer.is_full():
                self.buffer.free_buffer()
            block = self.buffer.read_blk_from_disk(file)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for j in range(0, len(integers) - 1, 2):
                tuples.append([integers[j], integers[j + 1]])
            for t in tuples:
                key = t[0]
                value = file
                if key not in hash_map_1.keys():
                    hash_map_1[key] = {value}
                else:
                    hash_map_1[key].add(value)

        for i in range(n2):
            file = 'relation_' + r2.lower() + '_' + str(i)
            if self.buffer.is_full():
                self.buffer.free_buffer()
            block = self.buffer.read_blk_from_disk(file)
            integers = self.buffer.read_blk_int(block)
            tuples = []
            for j in range(0, len(integers) - 1, 2):
                tuples.append([integers[j], integers[j + 1]])
            for t in tuples:
                key = t[0]
                value = file
                if key not in hash_map_2.keys():
                    hash_map_2[key] = {value}
                else:
                    hash_map_2[key].add(value)

        keys1 = list(hash_map_1.keys())
        keys2 = list(hash_map_2.keys())
        keys1.sort()
        keys2.sort()
        common_keys = [key for key in keys1 if key in keys2]
        B = self.buffer.num_all_blk
        old_IO = self.buffer.num_IO
        w_start, w_count = 0, 0
        results = []
        for key in common_keys:
            addr1s = list(hash_map_1[key])
            addr2s = list(hash_map_2[key])
            pointer = 0
            n1, n2 = len(addr1s), len(addr2s)
            # Read in the blocks and perform nested-loop join
            self.buffer.free_buffer()
            while pointer < n1:
                self.buffer.get_new_block()             # Apply a block for write buffer
                m = min(B - 2, n1 - pointer)
                for i in range(m):
                    self.buffer.read_blk_from_disk(addr1s[pointer])
                    pointer += 1
                for i in range(B - m - 2):              # Remain one block for relation 2
                    self.buffer.get_new_block()
                for j in range(n2):
                    self.buffer.read_blk_from_disk(addr2s[j])
                    t1_start = self.blk_size * 8
                    while t1_start < (m + 1) * self.blk_size * 8:
                        tuple1 = self.buffer.data[t1_start:t1_start + 64]
                        value1_1 = int("".join('%s' % p for p in tuple1[0:32]), 2)
                        value1_2 = int("".join('%s' % p for p in tuple1[32:64]), 2)

                        t2_start = 7 * self.blk_size * 8
                        while t2_start < B * self.blk_size * 8:
                            tuple2 = self.buffer.data[t2_start:t2_start + 64]
                            value2_1 = int("".join('%s' % p for p in tuple2[0:32]), 2)
                            value2_2 = int("".join('%s' % p for p in tuple2[32:64]), 2)

                            if value1_1 == value2_1 == key:
                                t = tuple1 + tuple2[32:64]
                                results.append([value1_1, value1_2, value2_2])

                                # Write to the output buffer
                                if w_start + 96 > self.blk_size * 8:
                                    rest_bit = self.blk_size * 8 - w_start
                                    self.buffer.data[w_start:self.blk_size * 8] = t[0:rest_bit]
                                    w_addr = 'hash_join_' + r1.lower() + '_' + r2.lower() + '_{index}'.format(
                                        index=w_count)
                                    w_count += 1
                                    self.buffer.write_blk_to_disk(0, w_addr)
                                    self.buffer.get_new_block()
                                    self.buffer.data[0:96 - rest_bit] = t[rest_bit:]
                                    w_start = rest_bit
                                else:
                                    self.buffer.data[w_start:w_start + 96] = t
                                    w_start += 96
                            t2_start += 64
                        t1_start += 64
                    self.buffer.free_blk(7)
                self.buffer.free_buffer()
        results.sort()
        print("IO times", self.buffer.num_IO - old_IO)
        return results

    def join(self, r1, r2, algorithm):
        if algorithm not in self.routing_table_join:
            return "No such algorithm!"
        return getattr(self, algorithm)(r1, r2)

    def check_join(self, r1, r2, query_result):
        n1 = self.relations[r1].num_block
        n2 = self.relations[r2].num_block

        self.buffer.free_buffer()
        integers1 = []
        for i in range(n1):
            addr1 = 'relation_' + r1.lower() + '_{index}'.format(index=i)
            block = self.buffer.read_blk_from_disk(addr1)
            integers1 += self.buffer.read_blk_int(block)
            self.buffer.free_blk(block)
        tuples1 = []
        for i in range(0, len(integers1) - 1, 2):
            tuples1.append([integers1[i], integers1[i + 1]])

        self.buffer.free_buffer()
        integers2 = []
        for i in range(n2):
            addr1 = 'relation_' + r2.lower() + '_{index}'.format(index=i)
            block = self.buffer.read_blk_from_disk(addr1)
            integers2 += self.buffer.read_blk_int(block)
            self.buffer.free_blk(block)
        tuples2 = []
        for i in range(0, len(integers2) - 1, 2):
            tuples2.append([integers2[i], integers2[i + 1]])

        results = []
        for t1 in tuples1:
            for t2 in tuples2:
                if t1[0] == t2[0]:
                    results.append([t1[0], t1[1], t2[1]])
        query_result.sort()
        results.sort()
        # print("\nChecking...")
        # print("The result should be")
        # for r in results:
        #     print(r)
        if query_result != results:
            print("Difference:")
            print([r for r in query_result if r not in results])
            print([r for r in results if r not in query_result])
        return query_result == results