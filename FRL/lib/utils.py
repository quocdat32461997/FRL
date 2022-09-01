# utils.py

import torch
import pickle
import random
import pandas as pd


def to_cuda(x):
    # Function to move torch.tensor either CPU or CUDA
    if torch.cuda.is_available():
        return x.to('cuda')
    else:
        return x.to('cpu')


def reindex_user_items(ratings):
    user_list = list(ratings['user'].drop_duplicates())
    user2id = {w: i for i, w in enumerate(user_list)}

    item_list = list(ratings['song_item'].drop_duplicates())
    item2id = {w: i for i, w in enumerate(item_list)}
    ratings['user'] = ratings['user'].apply(lambda x: user2id[x])
    ratings['song_item'] = ratings['song_item'].apply(lambda x: item2id[x])
    ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))

    return ratings, user2id, item2id, user_list, item_list


def leave_one_out(ratings):
  ratings['rank_latest'] = ratings.groupby(['user'])['song_item'].rank(method='first', ascending=True)
  test = ratings.loc[ratings['rank_latest'] == 1]
  train = ratings.loc[ratings['rank_latest'] > 1]
  assert train['user'].nunique()==test['user'].nunique()
  return train[['user', 'song_item', 'rating']], test[['user', 'song_item', 'rating']]


def negative_sampling(args, ratings, item_pool):
  interact_status = (ratings.groupby('user')['song_item'].apply(set).reset_index().rename(columns={'song_item': 'interacted_items'}))
  interact_status['non_interacted_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
  interact_status['random_non_interacted_samples'] = interact_status['non_interacted_items'].apply(lambda x: random.sample(x, args.num_ng_test))
  return interact_status[['user', 'non_interacted_items', 'random_non_interacted_samples']]


def get_rating_df(rating_file_path):
    for i in range(0, len(rating_file_path)):
        # print(i)
        with open(rating_file_path[3], 'rb') as f:
            # print(rating_file_path[i])
            rating_df = pickle.load(f)

        break

    df = pd.DataFrame(rating_df, columns=['user', 'song_item', 'rating'])
    # if i ==0:
    #   final_df = df
    #   break
    # else:
    #   final_df = pd.concat([final_df,df])
    # final_df.concat(df)

    # print(len(final_df))

    return df


# metrics

def hit(ng_item, pred_items):  # Hit Ratio
    if ng_item in pred_items:
        return 1
    return 0


def ndcg(ng_item, pred_items):  # Normalized Discounted cumulative gain
    if ng_item in pred_items:
        index = pred_items.index(ng_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    recommended_items = dict()

    for user, item, label in test_loader:
        user_id = user[0].item()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        recommended_items[user_id] = recommends

        ng_item = item[0].item()
        HR.append(hit(ng_item, recommends))
        NDCG.append(ndcg(ng_item, recommends))

    return np.mean(HR), np.mean(NDCG), recommended_items