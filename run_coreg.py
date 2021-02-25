from coreg.coreg import Coreg
import numpy as np

data_dir = "data/wells"
pred_file = "coreg_preds.csv"
k1 = 2
k2 = 4
p1 = 2
p2 = 5
max_iters = 120
pool_size = 200
verbose = True

# num_labeled = num_train + num_test
num_train = 118
num_test = 20
trials = 4

cr = Coreg(k1, k2, p1, p2, max_iters, pool_size)
cr.add_data(data_dir)

cr.run_trials(num_train, num_test, trials, verbose)

preds = cr.predict(cr.X_unlabeled)
np.savetxt(pred_file, preds, delimiter=",")

