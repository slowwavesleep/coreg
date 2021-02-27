from coreg.coreg import Coreg
import numpy as np

LABELED_PATH = "data/wells/Training_wells.csv"
UNLABELED_PATH = "data/wells/Empty_part.csv"
PREDICTIONS_PATH = "coreg_preds.csv"

k1 = [2, 3, 4]
k2 = [2, 3, 4]
p1 = [4, 1, 2]
p2 = [5, 2, 3]
max_iters = [100, 100, 100]
pool_size = [300, 300, 300]
random_state = [42, 13, 27]
weights_list_1 = ["uniform", "uniform", "uniform"]
weights_list_2 = ["uniform", "uniform", "uniform"]

verbose = True

# these are hard-coded values specific
# to a particular dataset
# meant to be generalized
# num_labeled = num_train + num_test
num_train = 110
num_test = 28
trials = 4

assert len(k1) == len(k2) == len(p1) == len(p2) == len(max_iters) == len(pool_size) == len(random_state)

preds = []
for k_neigh_1, k_neigh_2, power_1, power_2, cur_max_iters, cur_pool_size, cur_random_state, w1, w2 in zip(
        k1,
        k2,
        p1,
        p2,
        max_iters,
        pool_size,
        random_state,
        weights_list_1,
        weights_list_2):
    cr = Coreg(n_neighbors_1=k_neigh_1,
               n_neighbors_2=k_neigh_2,
               power_1=power_1,
               power_2=power_2,
               max_iters=cur_max_iters,
               pool_size=cur_pool_size,
               weights_1=w1,
               weights_2=w2)
    cr.add_data(LABELED_PATH, UNLABELED_PATH)

    cr.run_trials(num_train, num_test, trials, verbose, random_state=cur_random_state)

    cur_preds = cr.predict(cr.X_unlabeled)
    preds.append(cur_preds)

preds = np.mean(preds, axis=0)
np.savetxt(PREDICTIONS_PATH, preds, delimiter=",")
