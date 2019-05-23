from Buffer import *
if __name__ == '__main__':
    buf_size, blk_size = 520, 64
    buffer = Buffer(buf_size, blk_size)
    # print(buffer.blk_available)
    # print(buffer.data)

    blk = buffer.get_new_block()
    # print(buffer.blk_available)
    # print(buffer.data)

    for i in range(blk_size):
        buffer.data[blk + i] = 1

    print(buffer.blk_available)
    print(buffer.data)
    print(buffer.num_free_blk)

    buffer.write_blk_to_disk(0, 334455)
    print(buffer.blk_available)
    print(buffer.num_free_blk)

    buffer.get_new_block()
    buffer.get_new_block()

    print(buffer.blk_available)
    print(buffer.num_free_blk)

    buffer.read_blk_from_disk(blk)
    print(buffer.data)

    buffer.drop_blk_on_disk(334456)
