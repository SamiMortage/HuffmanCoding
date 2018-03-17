"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression

def list_of_tuples(dict_):
    ''' Return a sorted list of tuples using dict_,  where each tuple has the
    form (freq, symbol).

    @type dict_: dict(int, int)
                    the frequency dictionary
    @rtype: list[(int, int)]

    >>> dict_ = {4: 3, 5: 2, 6: 2}
    >>> list_of_tuples(dict_)
    [(2, 5), (2, 6), (3, 4)]
    '''
    list_of_tuples_ = sorted(((v, k) for k, v in dict_.items()), reverse=False)
    return list_of_tuples_


def adding_symbols(list_):
    ''' Return the root HuffmanNode of a Huffman Tree using the symbols and
    frequencies in list_.

    Precondition: list_ has at least 2 items.

    @type list_: list[(int, int)]
    @rtype: HuffmanNode

    >>> adding_symbols([(2, 65), (4, 66), (7, 67), ]) == HuffmanNode(None, \
HuffmanNode(None, HuffmanNode(65), HuffmanNode(66)), HuffmanNode(67))
    True
    '''
    for i in range(len(list_)):
        # Here we are making sure every 2nd element in a tuple is a HuffmanNode
        # to stay consistent throught the function and make sure the output is
        # correct.
        if not isinstance(list_[i][1], HuffmanNode):
            list_[i] = (list_[i][0], HuffmanNode(list_[i][1]))
    if len(list_) > 2:
        min_1 = list_[0][1]
        min_2 = list_[1][1]
        total_freq = list_[0][0] + list_[1][0]
        node = HuffmanNode(None, min_1, min_2)
        list_.pop(0)
        list_.pop(0)
        list_.append((total_freq, node))
        list_.sort()
        return adding_symbols(list_)
    # At this point, the list has only 2 items, which are the left and right \
    # children of the root node.
    else:
        tree = HuffmanNode(None, list_[0][1], list_[1][1])
    return tree


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dict_ = {}
    for byte in text:
        if byte not in dict_:
            dict_[byte] = 1
        else:
            dict_[byte] += 1
    return dict_


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq = {2: 6, 3: 4, 5: 10}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(2), \
HuffmanNode(3)), HuffmanNode(5))
    >>> result2 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
HuffmanNode(2)), HuffmanNode(5))
    >>> result3 = HuffmanNode(None, HuffmanNode(5), HuffmanNode(None, \
HuffmanNode(2), HuffmanNode(3)))
    >>> result4 = HuffmanNode(None, HuffmanNode(5), HuffmanNode(None, \
HuffmanNode(3), HuffmanNode(2)))
    >>> t == result1 or t == result2 or t == result3 or t == result4
    True
    >>> freq = {250: 7}
    >>> t = huffman_tree(freq)
    >>> t == HuffmanNode(None, HuffmanNode(250), HuffmanNode(-1)) or t == \
HuffmanNode(None, HuffmanNode(-1), HuffmanNode(250))
    True
    """
    # http://www.geeksforgeeks.org/greedy-algorithms-set-3-huffman-coding/
    # The website above gave us a clearer understanding of the algorithm.
    list_of_tuples_ = list_of_tuples(freq_dict)
    if len(list_of_tuples_) == 1:
        # The right child is just a place holder, it will not affect the code.
        return HuffmanNode(None, HuffmanNode(list_of_tuples_[0][1]),
                           HuffmanNode(-1))
    else:
        return adding_symbols(list_of_tuples_)


def get_leaves(tree):
    '''Return a list containing the symbols of leaves.

    @type tree: HuffmanNode
    @rtype: list[int]

    >>> a = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1),\
HuffmanNode(3)), HuffmanNode(2))
    >>> b = get_leaves(a)
    >>> b.sort()
    >>> b
    [1, 2, 3]
    '''
    if not tree.left and not tree.right:
        return [tree.symbol]
    else:
        if tree.left and tree.right:
            return get_leaves(tree.left) + get_leaves(tree.right)
        elif tree.left:
            return get_leaves(tree.left)
        else:
            return get_leaves(tree.right)


def find_code(int_, tree):
    '''Return the code corresponding to the symbol int in tree.

    @type int: int
    @type tree: HuffmanNode
    @rtype: str

    >>> a = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(3)), HuffmanNode(2))
    >>> find_code(2, a)
    '1'
    '''
    code = ""
    if tree.right and tree.left:
        if int_ in get_leaves(tree.left):
            code = '0'
            return code + find_code(int_, tree.left)
        elif int_ in get_leaves(tree.right):
            code = '1'
            return code + find_code(int_, tree.right)
    return code


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    leaves = get_leaves(tree)
    dict_ = {}
    for leaf in leaves:
        dict_[leaf] = find_code(leaf, tree)
    return dict_


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(None, \
HuffmanNode(10)))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.right.number
    1
    >>> tree.right.number
    2
    >>> tree.number
    3
    """
    list_ = find_internal(tree)
    for node in list_:
        node.number = list_.index(node)


def find_internal(tree):
    '''Return a list with the internal nodes of tree in postorder.

    @type tree: HuffmanNode
    @rtype: list[HuffmanNode]

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> find_internal(tree)
    [HuffmanNode(None, HuffmanNode(3, None, None), HuffmanNode(2, None, None)),\
 HuffmanNode(None, HuffmanNode(9, None, None), HuffmanNode(10, None, None)), \
HuffmanNode(None, HuffmanNode(None, HuffmanNode(3, None, None), \
HuffmanNode(2, None, None)), HuffmanNode(None, HuffmanNode(9, None, None), \
HuffmanNode(10, None, None)))]
    '''
    if tree is None or tree.is_leaf():
        return []
    else:
        return find_internal(tree.left) + find_internal(tree.right) + [tree]


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    code_dict = get_codes(tree)
    total_bits = 0
    for symbol in code_dict:
        total_bits += len(code_dict[symbol]) * freq_dict[symbol]
    total_symbols = 0
    for symbol in freq_dict:
        total_symbols += freq_dict[symbol]
    return total_bits / total_symbols


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10110000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    code = ''
    for byte in text:
        code += codes[byte]
    list_ = []
    str = ''
    for ch in code:
        str += ch
        if len(str) == 8:
            # max length of a byte in terms of bits has been reached, so we
            # need to append the current string to list_ and refresh the string.
            list_.append(bits_to_byte(str))
            str = ''
    # This means that the loop ended and thus you append the last str.
    list_.append(bits_to_byte(str))
    bytes_ = bytes(list_)
    return bytes_


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    internal = find_internal(tree)
    bytes_ = bytes([])
    for node in internal:
        if node.left.is_leaf():
            bytes_ += bytes([0])
            bytes_ += bytes([node.left.symbol])
        if not node.left.is_leaf():
            bytes_ += bytes([1])
            bytes_ += bytes([node.left.number])
        if node.right.is_leaf():
            bytes_ += bytes([0])
            bytes_ += bytes([node.right.symbol])
        if not node.right.is_leaf():
            bytes_ += bytes([1])
            bytes_ += bytes([node.right.number])
    return bytes_


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    >>> a = generate_tree_general(lst, 2)
    >>> a.number
    2
    """
    root = node_lst[root_index]
    left = HuffmanNode()
    right = HuffmanNode()
    if root.l_type is 1:  # It is an internal node
        left = generate_tree_general(node_lst, root.l_data)
    if root.l_type is 0:
        left = HuffmanNode(root.l_data)
    if root.r_type is 1:
        right = generate_tree_general(node_lst, root.r_data)
    if root.r_type is 0:
        right = HuffmanNode(root.r_data)
    root = HuffmanNode(None, left, right)
    root.number = root_index
    return root


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 4), \
    ReadNode(0, 5, 0, 6), ReadNode(0, 7, 0, 8), ReadNode(1, 0, 1, 0),\
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 5)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(1, None, None), HuffmanNode(2, None, None)), \
HuffmanNode(4, None, None)), HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(5, None, None), HuffmanNode(6, None, None)), \
HuffmanNode(None, HuffmanNode(7, None, None), HuffmanNode(8, None, None))))
    """
    root = node_lst[root_index]
    if root.l_type == 0 and root.r_type == 0:
        # In this case this node has two children that are leaves
        left = HuffmanNode(root.l_data)
        right = HuffmanNode(root.r_data)
        # We do this to keep track of which ReadNodes were used
        node_lst[root_index] = -1
    elif root.r_type == 0:
        # This means that the ReadNode in the index one before the root is
        # the left child.
        right = HuffmanNode(root.r_data)
        left = generate_tree_postorder(node_lst, root_index - 1)
    elif root.l_type == 0:
        # This means that the ReadNode in the index one before the root is
        # the right child
        right = generate_tree_postorder(node_lst, root_index - 1)
        left = HuffmanNode(root.l_data)
    else:
        # Here we have a root that has 2 children that are internal nodes. The
        # recursive call replaces every child on the right side of the root
        # with a -1 so we know the index of the first left child.
        left_index = ''  # Does not matter what this variable is at this point.
        right = generate_tree_postorder(node_lst, root_index - 1)
        for i in range(len(node_lst)):
            if node_lst[i] == -1:
                left_index = i - 1
                # Once it reaches a -1, we know that the left child is 1 before
                # it, so it ends.
                break
        left = generate_tree_postorder(node_lst, left_index)
    root = HuffmanNode(None, left, right)
    root.number = root_index
    return root


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))), HuffmanNode(None, \
HuffmanNode(4), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> result = generate_uncompressed(t, text, size)
    >>> result == bytes([5, 3, 1, 1])
    True
    >>> left = HuffmanNode(None, HuffmanNode(None, HuffmanNode(13), \
HuffmanNode(4)), HuffmanNode(5))
    >>> right = HuffmanNode(None, HuffmanNode(6), HuffmanNode(7))
    >>> tree = HuffmanNode(None, left, right)
    >>> text = bytes([9, 176])
    >>> size = 5
    >>> result = generate_uncompressed(tree, text, size)
    >>> result == bytes([13, 5, 4 , 6, 7])
    True
    """
    bytes_ = ''
    for byte in text:
        bytes_ += byte_to_bits(byte)
    codes = get_codes(tree)
    switch_keys = {y: x for x, y in codes.items()}
    list_ = []
    final_bytes = bytes([])
    str_ = ''
    for ch in bytes_:
        str_ += ch
        if str_ in switch_keys and len(list_) < size:
            list_.append(switch_keys[str_])
            str_ = ''
    final_bytes = bytes(list_)
    return final_bytes


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions
def get_huffman_leaves(tree):
    '''Return the leaves in tree.

    @type tree: HuffmanNode
    @rtype: list[HuffmanNode]

    >>> a = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1),\
HuffmanNode(3)), HuffmanNode(2))
    >>> b = get_huffman_leaves(a)
    >>> b
    [HuffmanNode(1, None, None), HuffmanNode(3, None, None), HuffmanNode(2, \
None, None)]
    '''
    if tree.is_leaf():
        return [tree]
    else:
        if tree.left and tree.right:
            return get_huffman_leaves(tree.left) + \
                   get_huffman_leaves(tree.right)
        elif tree.left:
            return get_huffman_leaves(tree.left)
        else:
            return get_huffman_leaves(tree.right)


def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    leaves = get_huffman_leaves(tree)
    code_dict_ = get_codes(tree)
    for i in range(len(leaves)):
        symbol = leaves[i].symbol
        leaves[i] = (leaves[i], len(code_dict_[symbol]), freq_dict[symbol])
        # This results in a list of tuples of the form (HuffmanNode, the length
        # of the code associated with that symbol, the number of times that
        # symbol occurs)
    for item1 in leaves:
        for item2 in leaves:
            # Compare every item to every other item in the list, if the code
            # length of one item is greater than another and it is more
            # frequent, then the tree will be improved if they switch places in
            # the tree because then the code with longer length and higher
            # frequency will have a shorter code length.
            if item1[1] > item2[1] and item1[2] > item2[2]:
                temp_ = item1[0].symbol
                item1[0].symbol = item2[0].symbol
                item2[0].symbol = temp_


if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
