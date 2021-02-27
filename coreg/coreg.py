from time import time
from collections import defaultdict

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from coreg.data_utils import load_data


class Coreg:
    """
    Instantiates a CoReg regressor.
    """
    def __init__(self,
                 n_neighbors_1: int = 3,
                 n_neighbors_2: int = 3,
                 power_1: int = 2,
                 power_2: int = 5,
                 max_iters: int = 100,
                 pool_size: int = 100,
                 n_jobs: int = 1,
                 weights_1: str = "uniform",
                 weights_2: str = "uniform"):

        self.n_neighbors_1, self.n_neighbors_2 = n_neighbors_1, n_neighbors_2  # number of neighbors
        self.power_1, self.power_2 = power_1, power_2  # distance metrics
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.n_neighbors_1,
                                      p=self.power_1,
                                      weights=weights_1,
                                      n_jobs=n_jobs)
        self.h2 = KNeighborsRegressor(n_neighbors=self.n_neighbors_2,
                                      p=self.power_2,
                                      weights=weights_2,
                                      n_jobs=n_jobs)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.n_neighbors_1,
                                           p=self.power_1,
                                           weights=weights_1,
                                           n_jobs=n_jobs)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.n_neighbors_2,
                                           p=self.power_2,
                                           weights=weights_2,
                                           n_jobs=n_jobs)

        self.is_fitted = False

        self.num_trials = None
        self.trial = None
        self.X = None
        self.y = None

    def add_data(self, labeled_path: str, unlabeled_path: str):
        """
        Adds data and splits into labeled and unlabeled.
        """
        self.X, self.y = load_data(labeled_path, unlabeled_path)

    def run_trials(self,
                   num_train: int = 100,
                   num_test: int = 100,
                   trials: int = 10,
                   verbose: bool = False,
                   random_state: int = 42):
        """
        Runs multiple trials of training.
        """
        self.num_trials = trials
        self.trial = 0
        self._initialize_storage()
        while self.trial < self.num_trials:
            t0 = time()
            print('Starting trial {}:'.format(self.trial + 1))            
            self.train(random_state=random_state,
                       num_labeled=num_train,
                       num_test=num_test,
                       verbose=verbose,
                       store_results=True)
            print('Finished trial {}: {:0.2f}s elapsed\n'.format(self.trial + 1, time() - t0))
            self.trial += 1
        self._set_best_estimator()
        self.is_fitted = True

    def train(self,
              random_state: int = -1,
              num_labeled: int = 100,
              num_test: int = 1000,
              verbose: bool = False,
              store_results: bool = False):
        """
        Trains the CoReg regressor.
        """
        start = time()
        self._split_data(random_state, num_labeled, num_test)
        self._fit_and_evaluate(verbose)
        if store_results:
            self._store_results(0)
        self._get_pool()
        if verbose:
            print('Initialized h1, h2: {:0.2f}s\n'.format(time() - start))
        for t in range(1, self.max_iters + 1):
            stop_training = self._run_iteration(t, start, verbose, store_results)
            if stop_training:
                if verbose:
                    print('Done in {} iterations: {:0.2f}s'.format(t, time() - start))
                break
        if verbose:
            print('Finished {} iterations: {:0.2f}s'.format(t, time() - start))

    def _run_iteration(self,
                       t: int,
                       start: float,
                       verbose: bool = False,
                       store_results: bool = False):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        if verbose:
            print('Started iteration {}: {:0.2f}s'.format(t, time() - start))
        self._find_points_to_add()
        added = self._add_points()
        if added:
            self._fit_and_evaluate(verbose)
            if store_results:
                self._store_results(t)
            self._remove_from_unlabeled()
            self._get_pool()
        else:
            stop_training = True
        return stop_training

    def _add_points(self):
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

    @staticmethod
    def _compute_delta(omega: np.array,
                       L_X: np.array,
                       L_y: np.array,
                       h: RegressorMixin,
                       h_temp: RegressorMixin):
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

    def _compute_deltas(self,
                        L_X: np.array,
                        L_y: np.array,
                        h: RegressorMixin,
                        h_temp: RegressorMixin):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        deltas = np.zeros((self.U_X_pool.shape[0],))
        for idx_u, x_u in enumerate(self.U_X_pool):
            # Make prediction
            x_u = x_u.reshape(1, -1)
            y_u_hat = h.predict(x_u).reshape(1, -1)
            # Compute neighbors
            omega = h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = np.vstack((L_X, x_u))
            y_temp = np.vstack((L_y, y_u_hat)) # use predicted y_u_hat
            h_temp.fit(X_temp, y_temp)
            delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
            deltas[idx_u] = delta
        return deltas

    def _evaluate_metrics(self, verbose: bool):
        """
        Evaluates KNN regressors on training and test data.
        """
        train1_hat = self.h1.predict(self.X_labeled)
        train2_hat = self.h2.predict(self.X_labeled)
        train_hat = 0.5 * (train1_hat + train2_hat)
        test1_hat = self.h1.predict(self.X_test)
        test2_hat = self.h2.predict(self.X_test)
        test_hat = 0.5 * (test1_hat + test2_hat)

        self.mse1_train = mean_squared_error(train1_hat, self.y_labeled, squared=False)
        self.mse1_test = mean_squared_error(test1_hat, self.y_test, squared=False)
        self.mse2_train = mean_squared_error(train2_hat, self.y_labeled, squared=False)
        self.mse2_test = mean_squared_error(test2_hat, self.y_test, squared=False)
        self.mse_train = mean_squared_error(train_hat, self.y_labeled, squared=False)
        self.mse_test = mean_squared_error(test_hat, self.y_test, squared=False)

        if verbose:
            print('MSEs:')
            print('  KNN1:')
            print('    Train: {:0.4f}'.format(self.mse1_train))
            print('    Test : {:0.4f}'.format(self.mse1_test))
            print('  KNN2:')
            print('    Train: {:0.4f}'.format(self.mse2_train))
            print('    Test : {:0.4f}'.format(self.mse2_test))
            print('  Combined:')
            print('    Train: {:0.4f}'.format(self.mse_train))
            print('    Test : {:0.4f}\n'.format(self.mse_test))

    def _find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}

        # Keep track of added idxs
        added_idxs = []
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y
            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y
            deltas = self._compute_deltas(L_X, L_y, h, h_temp)
            # Add largest delta (improvement)
            sort_idxs = np.argsort(deltas)[::-1] # max to min
            max_idx = sort_idxs[0]
            if max_idx in added_idxs:
                max_idx = sort_idxs[1]
            if deltas[max_idx] > 0:
                added_idxs.append(max_idx)
                x_u = self.U_X_pool[max_idx].reshape(1, -1)
                y_u_hat = h.predict(x_u).reshape(1, -1)
                self.to_add['x' + str(idx_h)] = x_u
                self.to_add['y' + str(idx_h)] = y_u_hat
                self.to_add['idx' + str(idx_h)] = self.U_idx_pool[max_idx]

    def _fit_and_evaluate(self, verbose: bool):
        """
        Fits h1 and h2 and evaluates metrics.
        """
        self.h1.fit(self.L1_X, self.L1_y)
        self.h2.fit(self.L2_X, self.L2_y)
        self._evaluate_metrics(verbose)

    def _get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_X_pool, self.U_y_pool, self.U_idx_pool = shuffle(self.U_X,
                                                                self.U_y,
                                                                range(self.U_y.size))
        self.U_X_pool = self.U_X_pool[:self.pool_size]
        self.U_y_pool = self.U_y_pool[:self.pool_size]
        self.U_idx_pool = self.U_idx_pool[:self.pool_size]

    def _initialize_storage(self):
        """
        Sets up metrics and estimators to be stored.
        """
        self.mses1_train = defaultdict(lambda: dict())
        self.mses1_test = defaultdict(lambda: dict())
        self.mses2_train = defaultdict(lambda: dict())
        self.mses2_test = defaultdict(lambda: dict())
        self.mses_train = defaultdict(lambda: dict())
        self.mses_test = defaultdict(lambda: dict())

        self.first_estimators = defaultdict(lambda: dict())
        self.second_estimators = defaultdict(lambda: dict())

    def _remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)
        self.U_y = np.delete(self.U_y, to_remove, axis=0)

    def _split_data(self,
                    random_state: int = -1,
                    num_train: int = 100,
                    num_test: int = 1000):
        """
        Shuffles data and splits it into train, test, and unlabeled sets.
        """
        test_end = num_train + num_test

        if random_state >= 0:
            X_shuffled, y_shuffled = shuffle(self.X[:test_end],
                                             self.y[:test_end],
                                             random_state=random_state)
        else:
            X_shuffled = self.X[:test_end]
            y_shuffled = self.y[:test_end]

        # Initial labeled, test, and unlabeled sets
        self.X_labeled = X_shuffled[:num_train]
        self.y_labeled = y_shuffled[:num_train]
        self.X_test = X_shuffled[num_train:test_end]
        self.y_test = y_shuffled[num_train:test_end]
        self.X_unlabeled = self.X[test_end:]
        self.y_unlabeled = self.y[test_end:]

        # Up-to-date training sets and unlabeled set
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]

    def _store_results(self, iteration):
        """
        Stores current MSEs.
        """
        self.mses1_train[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse1_train
        self.mses1_test[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse1_test
        self.mses2_train[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse2_train
        self.mses2_test[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse2_test
        self.mses_train[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse_train
        self.mses_test[f"trial_{self.trial}"][f"iter_{iteration}"] = self.mse_test

        self.first_estimators[f"trial_{self.trial}"][f"iter_{iteration}"] = self.h1
        self.second_estimators[f"trial_{self.trial}"][f"iter_{iteration}"] = self.h2

    def _set_best_estimator(self):
        result = []
        for key, value in self.mses_test.items():
            best_iter_num = min(value, key=value.get)
            best_iter_score = value[best_iter_num]
            result.append((key, best_iter_num, best_iter_score))
        best_trial, best_iter, _ = sorted(result, key=lambda item: item[2])[0]

        self.best_h1_score = self.mses1_test[best_trial][best_iter]
        self.best_h2_score = self.mses2_test[best_trial][best_iter]
        self.best_combined_score = self.mses_test[best_trial][best_iter]
        self.h1 = self.first_estimators[best_trial][best_iter]
        self.h2 = self.second_estimators[best_trial][best_iter]

    def predict(self, X):
        if self.is_fitted:
            h1_preds = self.h1.predict(X)
            h2_preds = self.h2.predict(X)
            return np.mean([h1_preds, h2_preds], axis=0)
        else:
            print("The model is not fitted!")


# TODO implement sklearn-style fit
