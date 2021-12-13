# preprocess.py

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import networkx as nx
from multiprocessing import Pool
from functools import partial

INFO_KEYS = ['id', 'title', 'artist_id', 'artist_name']


def main(args):

    # process rating data
    inter_data, users, songs = process_rating(args)

    # extract song entities and parse into knowledge-graph
    #process_kg(args)


    pass


def process_kg(args):
    # Function to map song entities in Million Song Dataset w/ interaction-data
    # and parse to knowledge graph

    print('converting to kg')

    # read and extract songs from mapped Echo Nest data
    files = []
    for sub in os.listdir(args.entity_data):
        if not sub.endswith('.DS_Store'):
            files.extend([os.path.join(args.entity_data, sub, file) for
                          file in os.listdir(os.path.join(args.entity_data, sub))])

    # parse into batches for parallel programming
    batch_size = len(files) // args.ncpu + 1
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    # extract song_id, title, artist_id, artist_naame, album
    pool = Pool(args.ncpu)
    data = pool.map(convert_kg, batches)

    # build knowledge graph
    graph, adj_mtrx = build_kg(data)

    # save graph and adj_mtrx
    with open(os.path.join(args.output_dir, 'graph.pkl'), 'wb') as file:
        pickle.dump(graph, file)
    with open(os.path.join(args.output_dir, 'adj_mtrx.pkl'), 'wb') as file:
        pickle.dump(adj_mtrx, file)
    pass


def process_rating(args):
    # function to read and process interaction-data
    # Args:
    #   - args: parsed command args
    #   - inter_data: np.array of interaction-data

    print('converting rating')

    # read inter-data
    with open(args.inter_data) as file:
        data = file.read().split('\n')[:1000000]

    # parse into batches for parallel programming
    batch_size = 100000 #len(data) // args.ncpu + 1
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    # extract unique users and songs
    pool = Pool(args.ncpu)
    data, users, songs = zip(*pool.map(convert_rating, batches))

    # flatten data and songs
    data = [x for d in data for x in d]
    songs = list(set([x for s in songs for x in s]))
    users = list(set([x for u in users for x in u]))

    # print meta-data
    print('Number of users', len(users))
    print('Number of songs', len(songs))

    # save binarized data and meta-data
    with open(os.path.join(args.output_dir, 'rating.pkl'), 'wb') as file:
        pickl.dump(data, file)
    with open(os.path.join(args.output_dir, 'meta_data.pkl'), 'wb') as file:
        pickle.dump({'users': users, 'songs': songs}, file)

    return data, users, songs


def convert_rating(data):
    # Function to binarize ratings by play-counts
    # in music, play-count 1 is considered as skip or no interest.
    # when play-count > 1, consider as interesting

    # split line by tab and get unique user and song ids
    songs, users = [], []
    for i, line in tqdm(zip(range(len(data)), data), total=len(data)):
        line = line.split('\t')  # split by tab
        line[-1] = int(line[-1])  # convert rating to int
        line[-1] = 0 if line[-1] == 0 else 1  # binarize rating
        data[i] = line
        songs.append(line[1])
        users.append(line[0])

    return data, list(set(users)), list(set(songs))


def build_kg(data):
    # new network graph
    graph = nx.DiGraph()

    for nodes in data:
        for node in nodes:
            # add edge: song-artist
            print(node)
            graph.add_edge(node['id'], node['artist_id'], relation='song.created_by')

            # update nodes: song and artist w/ real name
            graph.nodes[node['id']]['name'] = node['title']
            graph.nodes[node['artist_id']]['name'] = node['artist_name']

            # update
            # add edge: song-album if existing
            if node['album_name'] is not None and node['album_date'] is not None:
                graph.add_edge(node['id'], node['album_name'], relation='song.in_album') # song-album
                graph.add_edge(node['id'], node['album_date'], relation='song.album_released_in') # song-album-date
                graph.add_edge(node['album_name'], node['album_date'], relation='album.released_in') # album-release-date

    # return graph and adjacency-matrix as dict
    return graph, graph.adjacency()


def convert_kg(data):
    # Function to extract song info
    _data = []
    for i, file in tqdm(zip(range(len(data)), data), total=len(data)):
        # read file
        with open(file) as f:
            data[i] = json.load(f)['response']['songs']

        # skip if no data
        if len(data[i]) == 0:
            continue
        _data.append(data[i][0])

        # get album_name and album_data
        if len(_data[-1]['tracks']) == 0 or 'album_name' not in _data[-1]['tracks'][0] or \
                'album_date' not in _data[-1]['tracks'][0]:
            album_name = None
            album_date = None
        else:
            album_name = _data[-1]['tracks'][0]['album_name']
            album_date = _data[-1]['tracks'][0]['album_date'].split('-')[0]

        # get song_id, title, artist_id, artist_name,
        _data[-1] = {key: _data[-1][key] for key in INFO_KEYS}
        _data[-1]['album_name'] = album_name
        _data[-1]['album_date'] = album_date

    return _data


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
