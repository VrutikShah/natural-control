"""This file contains functions for utility."""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import collections
import argparse

logging.basicConfig(level=logging.INFO)

EDGES = ['oor', 'ool', 'oio', 'lt', 'rt', 'sp', 'chs', 'chr', 'chl', 'cf', 'iol', 'ior']


# B-0 oio K-0 [10,270,90];
# B-0 ool C-0 nt K-0 B-0 S [5,270,180];

def str2list(_str):
    _list = _str[1:-1].split(",")
    _list = list(map(int, _list))
    return _list


def split_triplet(triplet_str):
    elements = triplet_str.split(' ')
    first_node = []
    edge = []
    second_node = []
    attribute = ''
    dist_angle_list = str2list(elements[-1])
    elements = elements[:-1]
    if '(' in elements:
        start, end = elements.index('('), elements.index(')')
        attribute = " ".join(elements[start: end + 1])
        elements = elements[:start] + elements[end + 1:]
    for ele in elements:
        if ele in EDGES:
            edge.append(ele)
        elif not edge:
            first_node.append(ele)
        else:
            second_node.append(ele)
    assert len(edge) == 1, "edge should only contain one element {}".format(elements)
    return " ".join(first_node), " ".join(edge), " ".join(second_node), dist_angle_list


def convert_map(graph):
    tidied_map = collections.defaultdict(dict)
    for triplet in graph.strip().split(';'):
        triplet = triplet.strip()
        if not triplet:
            continue
        start, edge, end, _ = split_triplet(triplet)
        start, edge, end = start.strip(), edge.strip(), end.strip()
        tidied_map[start][edge] = end
    return tidied_map


def convert_map2(graph):
    tidied_map = collections.defaultdict(dict)
    for triplet in graph.strip().split(';'):
        triplet = triplet.strip()
        if not triplet:
            continue
        start, edge, end, dist_angle_list = split_triplet(triplet)
        start, edge, end = start.strip(), edge.strip(), end.strip()
        dist_angle_list.append(end)
        tidied_map[start][edge] = dist_angle_list
    return tidied_map

def parse_output(pred, tidied_map2):
    pred_list = pred.split(" ")
    output = []
    prev_out = ""
    for a in range(0, len(pred_list)-1, 2):
        node = pred_list[a]
        edge = pred_list[a+1]
        if node!=prev_out and a!=0:
            # print("prev out - ", prev_out, "curr-node", node)
            node = prev_out
        prev_out = tidied_map2[node][edge][-1]
        output.append(tidied_map2[node][edge][:-1])
    return output

def print_pred(preds=None):
    outputs = []
    if preds is None:    
        with open("../data/pred_test.txt", "r") as f:
            preds = f.readlines()
    with open("../data/custom.graph", "r") as f:
        graph = f.readline()
    tidied_map2 = convert_map2(graph)
    # print(tidied_map2)
    for line in preds:
        out = parse_output(line, tidied_map2)
        for outs in out:
            outputs.append("Dist={}, Angle={}, Final Angle={}".format(outs[0],outs[1],outs[2]))
        outputs.append("\n")
    with open("../data/custom.instructions", "w") as f:
        f.writelines("\n".join(outputs))



if __name__ == '__main__':
    print_pred()
    # parser = argparse.ArgumentParser(
    #     description='Test convert_map')
    # parser.add_argument('prediction_file', help='Prediction File')
    # args = parser.parse_args()
    # with open(args.prediction_file) as prediction_file:
    #     predictions = prediction_file.readlines()
    # map_dict = convert_map2(predictions)
    # print(map_dict)
