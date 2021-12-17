# train.py

import os
import time
import math
import torch
import random
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch.nn. as nn

from lib import *


def get_train_instance(train_df, negtaives_df):
    users, items, ratings = [], [], []
    train_ratings = pd.merge(train_df, negtaives_df[['user', 'non_interacted_items']], on='user')
    train_ratings['negatives'] = train_ratings['non_interacted_items'].apply(lambda x: random.sample(x, args.num_ng))
    for row in train_ratings.itertuples():
        users.append(int(row.user))
        items.append(int(row.song_item))
        ratings.append(float(row.rating))
        for i in range(args.num_ng):
            users.append(int(row.user))
            items.append(int(row.negatives[i]))
            ratings.append(float(0))

    dataset = Rating_Datset(user_list=users, item_list=items, rating_list=ratings)

    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

def get_test_instance(test_df, negtaives_df):
    users, items, ratings = [], [], []
    test_ratings = pd.merge(test_df, negtaives_df[['user', 'non_interacted_items']], on='user')
    for row in test_ratings.itertuples():
        users.append(int(row.user))
        items.append(int(row.song_item))
        ratings.append(float(row.rating))
        for i in getattr(row, 'non_interacted_items'):
            users.append(int(row.user))
            items.append(int(i))
            ratings.append(float(0))

    dataset = Rating_Datset(user_list=users, item_list=items, rating_list=ratings)

    return torch.utils.data.DataLoader(dataset, batch_size=args.num_ng_test + 1, shuffle=False, num_workers=4)


def main(args):
    # read processed data
    rating_file_path = []
    for files in os.listdir(args.data):
        file_name = os.path.join(args.data, files)
        rating_file_path.append(file_name)

    print('Rating files: {}'.format(rating_file_path))

    final_df = get_rating_df(rating_file_path)
    print('Data size: {}'.format(len(final_df)))
    final_df = final_df.iloc[:50000]

    # extract user-items df
    indexed_user_item_df, user2id, item2id, useer_list, item_list = reindex_user_items(final_df)

    # get and item pool
    user_pool = set(final_df['user'].unique())
    item_pool = set(final_df['song_item'].unique())

    # split data
    train_df, test_df = leave_one_out(indexed_user_item_df)

    # extract negative sampling
    negative_df = negative_sampling(indexed_user_item_df)

    # create train-data and test-data loaders
    train_loader = get_train_instance(train_df, negative_df)
    test_loader = get_test_instance(test_df, negative_df)
    print('Data loader: train {}, test {}'.
          format(type(train_loader), type(test_loader)))
    num_users = final_df['user'].nunique() + 1
    num_items = final_df['song_item'].nunique() + 1
    print('Number of users {} and items {}'.format(num_users, num_items))


    # create model and configure hyper-parameters and optimizer
    model = IRSModel(args, num_users, num_items)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    MODEL = 'model_ncf'
    best_hr = 0
    final_results = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_epoch = list()

        for user, item, label in train_loader:
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # print('loss/Train_loss', loss.item(), epoch)

        model.eval()
        print("average loss  for epoch number {}:".format(epoch))
        print(sum(loss_epoch) / len(loss_epoch))
        HR, NDCG, recommended_items = metrics(model, test_loader, args.top_k)
        final_results.append(recommended_items)
        print('Perfomance/HR@10', HR, epoch)
        print('Perfomance/NDCG@10', NDCG, epoch)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S",
                                                                                        time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            torch.save(model, '{}.pth'.format(MODEL))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))

    results_to_show = final_results[best_epoch - 1]
    print(len(results_to_show))

    print(results_to_show[0])

    stored_results_with_user_id = dict()
    for key, value in user2id.items():
        # print(key)
        # print(value)
        items = results_to_show[value]
        stored_results_with_user_id[key] = items

    len(stored_results_with_user_id)
    stored_results_with_user_id['c22d6e5efe41883b2770c68f6fabf55df5bbbad1']

    with open('ncf_model_recommendation.pkl', 'wb') as fb:
        pickle.dump(stored_results_with_user_id, fb)

    new_item2_id = dict()
    for key, value in item2id.items():
        new_item2_id[value] = key

    new_item2_id[0]

    final_res = dict()
    for key, val in stored_results_with_user_id.items():
        items_id = [(new_item2_id[item]) for item in val]
        final_res[key] = items_id

    final_res['c22d6e5efe41883b2770c68f6fabf55df5bbbad1']

    with open('ncf_model_recommendation_final.pkl', 'wb') as f:
        pickle.dump(final_res, f)

    with open('user2ids.pkl', 'wb') as f:
        pickle.dump(final_res, f)

    with open('item2ids.pkl', 'wb') as f:
        pickle.dump(item2id, f)

    pass


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/10k_songs/processed')
    parser.add_argument('--graph', type=str, default='dataset/10k_songs/graph.pkl')
    parser.add_argument("--k-hop", type=int, default=3, help="K-hop")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=4, help="training epoches")
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=64, help="predictive factors numbers in the model")
    parser.add_argument("--layers", nargs='+', default=[64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--num_ng", type=int, default=4, help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test", type=int, default=100, help="Number of negative samples for test set")
    parser.add_argument('-f')

    # execute w/ parsed args
    main(args = parser.parse_args())