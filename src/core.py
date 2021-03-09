import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


class AffinityPropagation:
    def __init__(self, iters_num=10):
        self.s = None
        self.shape = None
        self.a = None
        self.r = None
        self.labels = None
        self.clust_loc = None
        self.iters = iters_num

    def fit(self, csr_mat: sparse.csr_matrix):
        self.s = csr_mat
        self.shape = csr_mat.shape

        self.a = sparse.csr_matrix(self.shape)
        self.r = sparse.lil_matrix(self.shape)

        for _ in tqdm(range(self.iters)):
            self.__update_responsibility()
            self.__update_availability()
        self.__get_clusters()

    def predict(self, checkins: pd.DataFrame, user_id=None, n=10):
        if self.clust_loc is None:
            checkins["cluster"] = self.labels[checkins.user_id]
            self.clust_loc = checkins.groupby(by="cluster")["location_id"].value_counts()

        if user_id:
            target_cluster = self.labels[user_id]
            try:
                topn_locs_clusterwise = list(self.clust_loc[target_cluster][:n].index)
            except (IndexError, KeyError):
                return list(checkins["location_id"].value_counts().index[:n])
            topn_locs = topn_locs_clusterwise.copy()
        else:
            topn_locs = list(checkins["location_id"].value_counts().index[:n])

        return topn_locs

    def get_clusters(self):
        return self.clusters

    def __get_clusters(self):
        instances = (np.ravel(self.a.diagonal()) + np.ravel(self.r.diagonal())) > 0
        exemplars_idx = np.flatnonzero(instances)
        n_clusters = len(exemplars_idx)
        labels = np.ravel(self.s[:, exemplars_idx].argmax(-1))
        labels[exemplars_idx] = np.arange(n_clusters)  # getting cluster_num by  user_num
        clusters = [np.where(labels == i)[0] for i in range(n_clusters)]
        self.labels = labels
        self.clusters = clusters

    def __update_availability(self):
        a_temp = sparse.lil_matrix(self.r.shape)
        a = None
        r = self.r.copy()
        a_diag = np.zeros(self.r.shape[0])

        r_diag = np.ravel(r.diagonal())
        r.setdiag(0)
        r_pos = r.maximum(0).tocoo()  # getting positive mat
        r_sum = np.ravel(r_pos.sum(0))  # sum over columns

        for row, col, data in zip(r_pos.row, r_pos.col, r_pos.data):
            if col != row:
                a_diag[col] += data
            a = r_diag[col] + r_sum[col] - data
            a_temp[row, col] = a if a < 0 else 0

        self.a = a_temp.minimum(0).tocoo()
        self.a.setdiag(a_diag)

    def __update_responsibility(self):
        """
        Filling "self.r value"
        :return:
        """
        mat_as = self.a + self.s  # Sum availability and similarity
        mat_as_copy = mat_as.copy()  # matrix a + s (aux value)

        enum = np.arange(self.shape[0])
        max_ind = np.ravel(np.argmax(mat_as, -1))

        mat_as_copy[enum, max_ind] = -np.inf

        s_max = np.ravel(mat_as[enum, max_ind])
        rem_s_max = np.ravel(mat_as_copy.max(-1).todense())

        mat_as = mat_as.tocoo()  # we need this transformation to be able to iterate over rows and columns

        for row, col, data in zip(mat_as.row, mat_as.col, mat_as.data):
            self.r[row, col] = data - rem_s_max[row] if data == s_max[row] else data - s_max[row]
