import numpy as np
import time
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import data_utils


class Coreg():
    """
    Instantiates a CoReg regressor.
    """
    def __init__(self, k1=3, k2=3, p1=2, p2=5, max_iters=100, pool_size=100):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2 = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)

    def add_data(self, data_dir, random_state=-1, num_labeled=100,
                 num_test=1000):
        """
        Adds data and splits into labeled and unlabeled.
        """
        self.X, y = data_utils.load_data(data_dir)
        self.y = y.reshape(-1, 1)
        if random_state >= 0:
            self.X, self.y = shuffle(self.X, self.y, random_state=random_state)
        # Initial labeled, test, and unlabeled sets
        test_end = num_labeled + num_test
        self.X_labeled = self.X[:num_labeled]
        self.y_labeled = self.y[:num_labeled]
        self.X_test = self.X[num_labeled:test_end]
        self.y_test = self.y[num_labeled:test_end]
        self.X_unlabeled = self.X[test_end:]
        self.y_unlabeled = self.y[test_end:]
        # Up-to-date training sets and unlabeled set
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]

    def train(self, verbose=False):
        """
        Trains the CoReg regressor.
        """
        t0 = time.time()
        self.fit_and_evaluate(verbose)
        self.get_pool()
        if verbose:
            t1 = time.time()
            print 'Trained initial regressors: {:0.4f} seconds\n'.format(t1-t0)
        for t in range(self.max_iters):
            if verbose:
                t1 = time.time()
                print 'Started iteration {}: {:0.4f} seconds'.format(t, t1-t0)
            self.find_points_to_add()
            added = self.add_points()
            if added:
                self.fit_and_evaluate(verbose)
                self.remove_from_unlabeled()
                self.get_pool()
            else:
                if verbose:
                    t1 = time.time()
                    print 'Done in {} iterations: {:0.4f} seconds'.format(
                        t, t1-t0)
                break
        if verbose:
            t1 = time.time()
            print 'Finished all {} iterations: {:0.4f} seconds'.format(
                t+1, t1-t0)

    def add_points(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x1'] is not None:
            self.L2_X = np.vstack((self.L2_X, self.to_add['x1']))
            self.L2_y = np.vstack((self.L2_y, self.to_add['y1']))
            added = True
        if self.to_add['x2'] is not None:
            self.L1_X = np.vstack((self.L1_X, self.to_add['x2']))
            self.L1_y = np.vstack((self.L1_y, self.to_add['y2']))
            added = True
        return added

    def compute_delta(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X[idx_o].reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
        return delta

    def compute_deltas(self, L_X, L_y, h, h_temp):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        deltas = np.zeros((self.U_X_pool.shape[0],))
        for idx_u, x_u in enumerate(self.U_X_pool):
            # Make prediction
            x_u = x_u.reshape(1, -1)
            y_u = self.U_y_pool[idx_u].reshape(1, -1)
            y_u_hat = h.predict(x_u)
            # Compute neighbors
            omega = h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = np.vstack((L_X, x_u))
            y_temp = np.vstack((L_y, y_u))
            h_temp.fit(X_temp, y_temp)
            delta = self.compute_delta(omega, L_X, L_y, h, h_temp)
            deltas[idx_u] = delta
        return deltas

    def evaluate_metrics(self, verbose):
        """
        Evaluates KNN regressors on training and test data.
        """
        train1_hat = self.h1.predict(self.X_labeled)
        train2_hat = self.h2.predict(self.X_labeled)
        train_hat = 0.5 * (train1_hat + train2_hat)
        test1_hat = self.h1.predict(self.X_test)
        test2_hat = self.h2.predict(self.X_test)
        test_hat = 0.5 * (test1_hat + test2_hat)
        self.rmse1_train = np.sqrt(mean_squared_error(
            train1_hat, self.y_labeled))
        self.rmse1_test = np.sqrt(mean_squared_error(
            test1_hat, self.y_test))
        self.rmse2_train = np.sqrt(mean_squared_error(
            train2_hat, self.y_labeled))
        self.rmse2_test = np.sqrt(mean_squared_error(
            test2_hat, self.y_test))
        self.rmse_train = np.sqrt(mean_squared_error(
            train_hat, self.y_labeled))
        self.rmse_test = np.sqrt(mean_squared_error(
            test_hat, self.y_test))
        if verbose:
            print 'RMSEs:'
            print '  KNN1:'
            print '    Train: {:0.4f}'.format(self.rmse1_train)
            print '    Test: {:0.4f}'.format(self.rmse1_test)
            print '  KNN2:'
            print '    Train: {:0.4f}'.format(self.rmse2_train)
            print '    Test: {:0.4f}'.format(self.rmse2_test)
            print '  Combined:'
            print '    Train: {:0.4f}'.format(self.rmse_train)
            print '    Test: {:0.4f}\n'.format(self.rmse_test)

    def find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y
            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y
            deltas = self.compute_deltas(L_X, L_y, h, h_temp)
            # Add largest delta (improvement)
            if max(deltas) > 0:
                self.to_add['x' + str(idx_h)] = self.U_X_pool[np.argmax(
                    deltas)].reshape(1, -1)
                self.to_add['y' + str(idx_h)] = self.U_y_pool[np.argmax(
                    deltas)].reshape(1, -1)
                self.to_add['idx' + str(idx_h)] = self.U_idx_pool[np.argmax(
                    deltas)]

    def fit_and_evaluate(self, verbose):
        """
        Fits h1 and h2 and evalutes metrics.
        """
        self.fit_regressors()
        self.evaluate_metrics(verbose)

    def fit_regressors(self):
        """
        Fits h1 and h2.
        """
        self.h1.fit(self.L1_X, self.L1_y)
        self.h2.fit(self.L2_X, self.L2_y)

    def get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_X_pool, self.U_y_pool, self.U_idx_pool = shuffle(
            self.U_X, self.U_y, range(self.U_y.size))
        self.U_X_pool = self.U_X_pool[:self.pool_size]
        self.U_y_pool = self.U_y_pool[:self.pool_size]
        self.U_idx_pool = self.U_idx_pool[:self.pool_size]

    def remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)
        self.U_y = np.delete(self.U_y, to_remove, axis=0)