import numpy as np
from tqdm import tqdm

from src.core import AffinityPropagation
from src.utils import load_edges, load_checkins

EDGES_PATH = "data/Gowalla_edges.txt"
CHECKINS_PATH = "data/Gowalla_totalCheckins.txt"


def get_intersect(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred))


if __name__ == '__main__':
    edges = load_edges(EDGES_PATH)
    checkins_train, checkins_test = load_checkins(CHECKINS_PATH)

    ap = AffinityPropagation(10)
    ap.fit(edges)
    topn_base = ap.predict(checkins_train)
    cluster_prec = 0
    base_prec = 0
    norm = len(np.unique(checkins_test.user_id)) * 10
    for user_id in tqdm(np.unique(checkins_test.user_id)):
        gt_locs = checkins_test.loc[checkins_test.user_id == user_id, "location_id"].values
        if user_id in checkins_train.user_id:
            train_locs = ap.predict(checkins_train, user_id)
            cluster_prec += get_intersect(gt_locs, train_locs)
            base_prec += get_intersect(gt_locs, topn_base)

    print(f"Base AP {base_prec / norm}")
    print(f"Cluster AP {cluster_prec / norm}")
