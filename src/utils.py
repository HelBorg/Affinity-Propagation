import numpy as np
import pandas as pd
import scipy.sparse
from sklearn import model_selection


def load_edges(path):
    edges = np.loadtxt(path, dtype=int)
    users = np.unique(edges)
    users_num = np.max(users) + 1
    data_mat = sparse.coo_matrix(([1] * len(edges), (edges[:, 0], edges[:, 1])), shape=(users_num, users_num))
    return data_mat


def load_checkins(path):
    names = ["user_id", "check-in time", "latitude", "longitude", "location_id"]
    checkins_df = pd.read_csv(path, sep="\t", header=None, names=names)
    users = np.unique(checkins_df.User_ID)

    train_users, test_users = model_selection.train_test_split(users, test_size=0.01, shuffle=True)

    train = checkins_df.loc[checkins_df['user_id'].isin(train_users)]
    test = checkins_df.loc[checkins_df['user_id'].isin(test_users)]

    return train, test
