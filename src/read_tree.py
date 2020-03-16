import pdb
import os
import sys
import math
import warnings
import argparse
from intervaltree import Interval, IntervalTree


CC_KEY = ["and", "or", "but", "nor", "and\/or"]
CC_SEP = [",", ";", ":"]

def _parse_tree(text, left_bracket='(', right_bracket=')'):
    stack = []
    _buffer = []
    for line in [text] if isinstance(text, str) else text:
        line = line.lstrip()
        if not line:
            continue
        for char in line:
            if char == left_bracket:
                stack.append([])
            elif char == ' ' or char == '\n':
                if _buffer:
                    stack[-1].append(''.join(_buffer))
                    _buffer = []
            elif char == right_bracket:
                if _buffer:
                    stack[-1].append(''.join(_buffer))
                    _buffer = []
                if len(stack) > 1:
                    stack[-2].append(stack.pop())
                else:
                    yield stack.pop()
            else:
                _buffer.append(char)


class Coordination(object):
    __slots__ = ('cc', 'conjuncts', 'seps', 'label')

    def __init__(self, cc, conjuncts, seps=None, label=None):
        assert isinstance(conjuncts, (list, tuple)) and len(conjuncts) >= 2
        assert all(isinstance(conj, tuple) for conj in conjuncts)
        conjuncts = sorted(conjuncts, key=lambda span: span[0])
        # NOTE(chantera): The form 'A and B, C' is considered to be coordination.  # NOQA
        # assert cc > conjuncts[-2][1] and cc < conjuncts[-1][0]
        assert cc > conjuncts[0][1] and cc < conjuncts[-1][0]
        if seps is not None:
            if len(seps) == len(conjuncts) - 2:
                for i, sep in enumerate(seps):
                    assert conjuncts[i][1] < sep and conjuncts[i + 1][0] > sep
            else:
                warnings.warn(
                    "Coordination does not contain enough separators. "
                    "It may be a wrong coordination: "
                    "cc={}, conjuncts={}, separators={}"
                    .format(cc, conjuncts, seps))
        else:
            seps = []
        self.cc = cc
        self.conjuncts = tuple(conjuncts)
        self.seps = tuple(seps)
        self.label = label

    def get_pair(self, index, check=False):
        pair = None
        for i in range(1, len(self.conjuncts)):
            if self.conjuncts[i][0] > index:
                pair = (self.conjuncts[i - 1], self.conjuncts[i])
                assert pair[0][1] < index and pair[1][0] > index
                break
        if check and pair is None:
            raise LookupError(
                "Could not find any pair for index={}".format(index))
        return pair

    def __repr__(self):
        return "Coordination(cc={}, conjuncts={}, seps={}, label={})".format(
            self.cc, self.conjuncts, self.seps, self.label)

    def __eq__(self, other):
        if not isinstance(other, Coordination):
            return False
        return self.cc == other.cc \
            and len(self.conjuncts) == len(other.conjuncts) \
            and all(conj1 == conj2 for conj1, conj2
                    in zip(self.conjuncts, other.conjuncts))


def _extract(tree):
    words = []
    postags = []
    spans = {}
    coords = {}

    def _traverse(tree, index):
        begin = index
        label = tree[0].split("-")[0]
        if len(tree) == 2 and isinstance(tree[1], str):  # Leaf
            words.append(tree[1])
            postags.append(label)
        else:  # Node
            conjuncts = []
            cc = None
            for child in tree[1:]:
                child_label = child[0]
                assert child_label not in ["-NONE-", "``", "''"]
                child_span = _traverse(child, index)
                if "COORD" in child_label:
                    conjuncts.append(child_span)
                elif child_label == "CC" \
                    or (child_label.startswith("CC-")
                        and child_label != "CC-SHARED"):
                    assert isinstance(child[1], str)
                    cc = child_span[0]
                index = child_span[1] + 1
            if cc is not None and len(conjuncts) >= 2:
                seps = []
                if len(conjuncts) > 2:
                    # find separators
                    for i in range(1, len(conjuncts) - 1):
                        sep = _find_separator(words,
                                              conjuncts[i - 1][1] + 1,
                                              conjuncts[i][0],
                                              search_len=2)
                        if sep is None:
                            warnings.warn(
                                "Could not find separator: "
                                "left conjunct={}, right conjunct={}, "
                                "range: {}".format(
                                    conjuncts[i - 1], conjuncts[i],
                                    words[conjuncts[i - 1][0]:
                                          conjuncts[i][1] + 1]))
                            continue
                        seps.append(sep)
                coords[cc] = Coordination(cc, conjuncts, seps, label)
            index -= 1
        span = (begin, index)
        if span not in spans:
            spans[span] = [label]
        else:
            spans[span].append(label)
        return span

    _traverse(tree[0] if len(tree) == 1 else tree, index=0)
    return words, postags, spans, coords

def _find_separator(words, search_from, search_to, search_len=2):
    """
    NOTE: `search_from` is inclusive but `search_to` is not inclusive
    """
    assert search_len > 1
    diff = search_to - search_from
    if diff < 1:
        return None
    half = math.ceil(diff / 2)
    if half < search_len:
        search_len = half
    for i in range(search_len):
        if words[search_to - 1 - i].lower() in CC_KEY:
            return search_to - 1 - i
        elif words[search_from + i].lower() in CC_KEY:
            return search_from + i
    for i in range(search_len):
        if words[search_to - 1 - i] in CC_SEP:
            return search_to - 1 - i
        elif words[search_from + i] in CC_SEP:
            return search_from + i
    return None

def find_depth(it_tree, node):
    # find number of nodes above this node
    overlap_nodes = it_tree.overlap(node.conjuncts[0][0], node.conjuncts[-1][1])
    # remove those which are a subset 
    overlap_nodes = list(filter(lambda x: node.conjuncts[0][0]>=x[0] and node.conjuncts[-1][1]<=x[1], list(set(overlap_nodes))))
    return len(overlap_nodes) # atleast one - which is the node itself

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp')
    parser.add_argument('--out_fp')
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    inp_file = open(args.inp_fp,'r')
    out_file = open(args.out_fp,'w')
    trees = list(_parse_tree(inp_file))
    total_conjunct_sents, lower_conjunct_sents = 0, 0
    total_conjuncts, lower_conjuncts = 0, 0
    depths = dict()
    for tree in trees:
        it = IntervalTree()
        words, postags, spans, coords = _extract(tree)
        if len(coords) == 0:
            continue
        total_conjuncts += len(coords)
        total_conjunct_sents += 1
        coord_spans = []
        lower_conjunct = False
        for coord in coords.values():
            it[coord.conjuncts[0][0]:coord.conjuncts[-1][1]] = coord
        
        add_words = words.copy()
        for coord in coords.values():
            depth = find_depth(it, coord)
            add_words[coord.conjuncts[0][0]] = '[D%d] '%(depth)+add_words[coord.conjuncts[0][0]]
            add_words[coord.conjuncts[-1][1]] = add_words[coord.conjuncts[-1][1]]+' [D%d] '%(depth)
            for conjunct_num, conjunct in enumerate(coord.conjuncts):
                if conjunct_num == len(coord.conjuncts)-1:
                    continue
                add_words[conjunct[1]] = add_words[conjunct[1]]+' [S]'
            if depth not in depths:
                depths[depth] = 0
            depths[depth] += 1
        add_sent = ' '.join(add_words)
        out_file.write(add_sent+'\n')

    inp_file.close()
    out_file.close()
    print(depths)
    print('Total sentences = %d'%len(trees))
    print('Total Conjunctive sentences = %d'%total_conjunct_sents)
    print('Total Conjuncts = %d'%total_conjuncts)

    return

if __name__ == '__main__':
    main()