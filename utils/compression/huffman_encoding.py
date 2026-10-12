"""Huffman coding of a codebook-index stream.

Only the coded length is needed for the compression ratio, so this builds the
Huffman tree and returns the total number of bits without emitting the stream.
"""

import heapq
import itertools

import numpy as np


class Node:
    """Node of a Huffman tree."""

    def __init__(self, probability, index, left=None, right=None):
        self.probability = probability
        self.index = index
        self.left = left
        self.right = right
        self.code = ''


class HuffmanEncode:
    def __init__(self, bits=5):
        self.codes = {}
        self.initial_bits = bits

    def frequency(self, data):
        """Distinct symbols in `data` (non-negative ints) and their counts."""
        counts = np.bincount(np.asarray(data))
        indices = np.flatnonzero(counts)
        return indices, counts[indices]

    def codify(self, node, value=''):
        """Assign a code to every leaf by walking the tree."""
        new_value = value + str(node.code)
        if node.left:
            self.codify(node.left, new_value)
        if node.right:
            self.codify(node.right, new_value)
        if not node.left and not node.right:
            self.codes[node.index] = new_value
        return self.codes

    def get_gain(self, data, coding, counts):
        """Total coded size of `data` in bits, and the average bits per symbol."""
        after = 0
        for symbol in coding.keys():
            after += counts[symbol] * len(coding[symbol])
        return after, after / len(data)

    @classmethod
    def encode(cls, data, bits=5):
        """Huffman-code `data`; returns (total bits, average bits per symbol)."""
        huffman = cls(bits=bits)
        symbols, frequencies = huffman.frequency(data)

        # Min-heap of (count, insertion order, node): always merge the two rarest nodes.
        counter = itertools.count()
        heap = [(f, next(counter), Node(f, s)) for s, f in zip(symbols, frequencies)]
        heapq.heapify(heap)
        while len(heap) > 1:
            f_right, _, right = heapq.heappop(heap)
            f_left, _, left = heapq.heappop(heap)
            left.code = 0
            right.code = 1
            new = Node(f_left + f_right, left.index + right.index, left, right)
            heapq.heappush(heap, (new.probability, next(counter), new))

        coding = huffman.codify(heap[0][2])
        counts = dict(zip(symbols, frequencies))
        return huffman.get_gain(data, coding, counts)
