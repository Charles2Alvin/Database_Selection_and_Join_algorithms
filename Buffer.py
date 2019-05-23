import os


class Buffer(object):
    """
    A buffer that can read and write blocks in the external memory.

    Attributes:
            num_IO: Number of IOs.
            buf_size: Buffer size (bytes).
            blk_size: Block size (bytes).
            num_all_blk: Number of blocks that can be kept in the buffer.
            num_free_blk: Number of available blocks in the buffer.
            data: Data of the buffer.
    """
    def __init__(self, buf_size, blk_size):
        self.num_IO = 0
        self.buf_size = buf_size
        self.blk_size = blk_size
        self.num_all_blk = buf_size // blk_size
        self.num_free_blk = self.num_all_blk
        self.data = [0] * buf_size * 8
        self.blk_available = [True] * self.num_all_blk

    def is_full(self):
        return self.num_free_blk == 0

    def get_new_block(self):
        """
        Apply for a new block from the buffer.

        Returns:
            False if no free blocks are available in the buffer;
            otherwise the index of the block.
        """
        if self.num_free_blk == 0:
            print("Buffer is full!")
            return False
        for i in range(self.num_all_blk):   # find the first available block
            if self.blk_available[i] is True:
                new_blk = i
                break
        self.num_free_blk -= 1
        self.blk_available[new_blk] = False
        return new_blk

    def write_blk_to_disk(self, blk_index, addr):
        """ Write a block in the buffer to the hard disk by the index of the block """
        path = './blk_data/{name}.txt'.format(name=addr)
        with open(path, 'w') as f:
            start = blk_index * self.blk_size * 8
            for i in range(start, start + self.blk_size * 8):
                f.write(str(self.data[i]))

        # Record the IO changes
        # If writing a free block to disk, don't increase free blocks
        self.num_IO += 1
        if not self.blk_available[blk_index]:
            self.num_free_blk += 1
        self.blk_available[blk_index] = True

    def read_blk_from_disk(self, filename, read_start=0):
        """
        Read a block from the hard disk to the buffer by the address of the block
        Args:
            filename: The filename of the block data
            read_start: The position of bits from which you wants to start reading
        """
        # print("Read a block from a disk file, starting from bit", read_start)
        if self.num_free_blk == 0:
            print("Buffer overflows!")
            return False

        for i in range(self.num_all_blk):   # find the first available block
            if self.blk_available[i]:
                blk = i
                break

        self.blk_available[blk] = False
        self.num_free_blk -= 1
        self.num_IO += 1

        path = './blk_data/{name}.txt'.format(name=filename)
        with open(path, 'r') as f:
            file_data = f.read()
            file_data = file_data[read_start: read_start + self.blk_size * 8]
            # print(len(file_data), read_start)
            start = blk * self.blk_size * 8
            for i in range(self.blk_size * 8):  # write a block's content into the free block in the buffer
                self.data[start + i] = int(file_data[i])
        return blk

    def free_buffer(self):
        """ Free up the memory in the buffer """
        self.num_free_blk = self.num_all_blk
        self.blk_available = [True] * self.num_all_blk

    def free_blk(self, blk_index):
        """ Release the memory in a given block """
        self.blk_available[blk_index] = True
        self.num_free_blk += 1

    @staticmethod
    def drop_blk_on_disk(addr):
        """ Remove a block file on the disk which is specified by the address. """
        dir = os.path.dirname(__file__)     # current directory
        file = '/blk_data/{name}.txt'.format(name=addr)
        path = dir + file
        if os.path.exists(path):
            os.remove(path)
            print("File {name} successfully removed".format(name=file))
        else:
            print("No such file: {file}".format(file=path))

    def read_buf_int(self):
        integers = []
        start, end = 0, 32
        while end <= self.buf_size * 8:
            number = self.data[start:end]
            number = int("".join('%s' % i for i in number), 2)
            integers.append(number)
            start += 32
            end += 32
        return integers

    def read_blk_int(self, blk):
        integers = []
        start = blk * self.blk_size * 8
        end = start + 32
        while start < (blk + 1) * self.blk_size * 8:
            number = self.data[start:end]
            number = int("".join('%s' % i for i in number), 2)
            integers.append(number)
            start += 32
            end += 32
        return integers

    def write_buf_to_disk(self, filename):
        """ Write the whole buffer to the hard disk """
        path = './blk_data/{name}.txt'.format(name=filename)
        with open(path, 'w') as f:
            for i in range(len(self.data)):
                f.write(str(self.data[i]))

        # Record the IO changes
        self.num_IO += 8
        self.free_buffer()

    def read_buf_from_disk(self, filename):
        """ Read to the whole buffer from the hard disk """
        path = './blk_data/{name}.txt'.format(name=filename)
        with open(path, 'r') as f:
            data = f.read()
            for i in range(len(data)):
                self.data[i] = data[i]

        # Record the IO changes
        self.num_IO += 8
        self.num_free_blk = 0
        self.blk_available = [False] * self.num_all_blk

    def find_min(self, blk, byte_per_tuple, tuple_pointer):
        """
        Find the tuple with min values in the selected block
        :param blk: The index of the selected block
        :param byte_per_tuple: How many bytes a tuple would take
        :return: The bits of the selected tuples and the index of the block which holds the tuple
        """
        start = blk * self.blk_size * 8
        end = start + self.blk_size * 8
        min_value = 1000000
        min_tuple = []
        num_tuple = (end - start) // (byte_per_tuple * 8)
        for i in range(tuple_pointer, num_tuple):
            t_start = start + i * byte_per_tuple * 8
            t_end = t_start + byte_per_tuple * 8
            cur_tuple = self.data[t_start: t_end]
            value1 = int("".join('%s' % j for j in cur_tuple[0:32]), 2)
            value2 = int("".join('%s' % j for j in cur_tuple[32:64]), 2)
            if value1 < min_value:
                min_tuple = cur_tuple.copy()
                min_value = value1
            # print(t_start, t_end, value1, value2, min_value)
            t_start += byte_per_tuple * 8
            t_end += byte_per_tuple * 8

        return min_tuple, min_value






