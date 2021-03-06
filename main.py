from src import utils

EDGES_PATH = "data/Gowalla_edges.txt"
CHECKINS_PATH = "data/Gowalla_totalCheckins.txt"

if __name__ == '__main__':
    edge_mat = utils.load_edges(EDGES_PATH)
    train_checkins, test_checkins = utils.load_checkins(CHECKINS_PATH)
