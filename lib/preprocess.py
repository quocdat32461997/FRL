# preprocess.py

import os
import pickle
import argparse
import numpy as np
import networkx as nx
from multiprocessing import Pool
from functools import partial


def main(args):

    # process rating data
    inter_data = process_rating(args)

    # extract song entities and parse into knowledge-graph
    process_kg(args, inter_data)


    pass


def process_kg(args, data):
    # Function to map song entities in Million Song Dataset w/ interaction-data
    # and parse to knowledge graph

    # TODO: read and extract songs from MSD dataset

    # add song entities from interaction_data into knowledge graph
    pass


def process_rating(args):
    # function to read and process interaction-data
    # Args:
    #   - args: parsed command args
    #   - inter_data: np.array of interaction-data

    print('converting rating')

    # read inter-data
    with open(args.inter_data) as file:
        inter_data = file.read().split('\n')

    # parse into batches for parallel programming
    batch_size = len(inter_data) // args.ncpu + 1
    batches = [inter_data[i:i + batch_size] for i in range(len(data), batch_size)]

    # parse into partial function
    _convert_rating = partial(convert_rating)

    # extract unique users and songs
    pool = Pool(ags.ncpu)
    inter_data = pool.map(_convert_rating, batches)
    inter_data = np.array([x for x in inter_data])

    with open(os.path.join(args.output_dir, 'rating.txt'), 'w') as file:
        pickle.dump(inter_data, file)

    return inter_data


def convert_rating(data):
    # Function to binarize ratings by play-counts
    # in music, play-count 1 is considered as skip or no interest.
    # when play-count > 1, consider as interesting

    # split line by tab and get unique user and song ids
    for i, line in enumerate(data):
        line = line.split('\t')  # split by tab
        line[-1] = int(line[-1])  # convert rating to int
        line[-1] = 0 if line[-1] == 0 else 1  # binarize rating
        data[i] = line

    return data


def convert_kg(args):
    print('converting to kg')

    pass


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--inter-data', type=str, required=True)
    parser.add_argument('--entity-data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=1)

    # execute main line
    main(parser.parse_args())
