"""
Not updated yet.
"""

import scipy.optimize
import pykoopman as pk
import numpy as np
np.random.seed(42) # for reproducibility
import h5py
import hydra
import mlflow
import logging
import scipy
import dask
import sys
from pathlib import Path
from config import train_test_dmd_config_class, edmdc_training_class, dmdc_training_class
from utils.hydra_mlflow_decorator import log_hydra_to_mlflow
from filepaths import filepath_dataset_from_name, dir_current_hydra_output


import warnings
warnings.filterwarnings("ignore")

class normalizer:
    def __init__(self, data):
        self.mean = np.mean(data, axis=(0, 2))
        self.std = np.std(data, axis=(0, 2))
    
    def normalize(self, data):
        return (data - self.mean[:, None]) / self.std[:, None]
    
    def denormalize(self, data):
        return data * self.std[:, None] + self.mean[:, None]    

def convert_tensors(states, controls):
    x = states[:, :, :-1]
    xp1 = states[:, :, 1:]
    u = controls[:, :, :-1]

    # concatenate timesteps
    X = np.zeros((x.shape[1], x.shape[0] * x.shape[2]))
    Xp1 = np.zeros((x.shape[1], x.shape[0] * x.shape[2]))
    U = np.zeros((u.shape[1], u.shape[0] * u.shape[2]))
    for i in range(x.shape[0]):
        X[:, i * x.shape[2]:(i + 1) * x.shape[2]] = x[i]
        Xp1[:, i * x.shape[2]:(i + 1) * x.shape[2]] = xp1[i]
        U[:, i * u.shape[2]:(i + 1) * u.shape[2]] = u[i]
    
    return X.T, Xp1.T, U.T

def fit_edmdc(x, cfg: edmdc_training_class, states, controls, test = False, states_test=None, controls_test=None):
    logging.info('fitting with parameters: {}'.format(x))
    # determine if on windows
    if 'win' in sys.platform:
        print('fitting with parameters: {}'.format(x))

    # select trajectory
    if test is False and type(cfg) == edmdc_training_class:
        idx = np.random.choice(states.shape[0], cfg.n_samples_train)
        states, controls = states[idx], controls[idx]
    elif test is False and type(cfg) == dmdc_training_class:
        if cfg.n_samples_train is not None:
            logging.info('selecting {} samples for training'.format(cfg.n_samples_train))
            idx = np.random.choice(states.shape[0], cfg.n_samples_train)
            states, controls = states[idx], controls[idx]
    X, Xp1, U = convert_tensors(states, controls)

    if type(cfg) == edmdc_training_class:
        c1, c2, n_centers, kernel_width, include_states = x
        n_centers = int(n_centers) 
        include_states = bool(include_states)
        # create model
        EDMDc = pk.regression.EDMDc()
        centers = np.random.uniform(c1, c2, (states.shape[1], n_centers))
        RBF = pk.observables.RadialBasisFunction(
            rbf_type="thinplate",
            n_centers=centers.shape[1],
            centers=centers,
            kernel_width=kernel_width,
            polyharmonic_coeff=1.0,
            include_state=include_states,
        )
        model = pk.Koopman(observables=RBF, regressor=EDMDc)
    elif type(cfg) == dmdc_training_class:
        DMDc = pk.regression.DMDc(svd_rank=x[0] + controls.shape[1],
                                  svd_output_rank=x[0],
        )
        model = pk.Koopman(regressor=DMDc)
    # fit model
    model.fit(X, y=Xp1, u=U)

    # calculate fitness (mse)
    mse_train = calculate_mse(states, controls, model)
    rmse_train = np.sqrt(mse_train)
    mse = calculate_mse(states_test, controls_test, model)
    rmse = np.sqrt(mse)

    if type(cfg) == edmdc_training_class:
        # log to mlflow
        # mlflow.log_metric('mse', mse)
        # mlflow.log_metric('rmse', rmse)
        logging.info('MSE: {} / RMSE: {} for c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(mse, rmse, c1, c2, n_centers, kernel_width, include_states))
        if 'win' in sys.platform:
            print('MSE: {} / RMSE: {} for c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(mse, rmse, c1, c2, n_centers, kernel_width, include_states))
    if type(cfg) == dmdc_training_class:
        # log to mlflow
        # mlflow.log_metric('mse', mse)
        # mlflow.log_metric('rmse', rmse)
        logging.info('Train: MSE: {} / RMSE: {} for rank: {}'.format(mse_train, rmse_train, x[0]))
        logging.info('Test: MSE: {} / RMSE: {} for rank: {}'.format(mse, rmse, x[0]))
        if 'win' in sys.platform:
            print('MSE: {} / RMSE: {} for rank: {}'.format(mse, rmse, x[0]))
    # print('MSE: {} / RMSE: {} for c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(mse, rmse, c1, c2, n_centers, kernel_width, include_states))
    return mse

def calculate_mse(states, controls, model):
    mse_list = []
    for i in range(states.shape[0]):
        Xtrue = states[i, :, :]
        Utrue = controls[i, :, :-1]

        Xtrue0 = Xtrue[:, 0]

        Xkoop = model.simulate(Xtrue0, Utrue.T, n_steps=Xtrue.shape[1]-1)

        mse = np.mean((Xtrue[:,:-1].T - Xkoop)**2)
        mse_list.append(mse)
    mse = np.mean(np.array(mse_list))
    return mse

@log_hydra_to_mlflow
def train_edmdc(cfg: train_test_dmd_config_class):
    logging.info('starting edmdc training')
    logging.info('train test dmd config: {}'.format(cfg))

    # Load data
    path = filepath_dataset_from_name(cfg.dataset_name)
    dataset = h5py.File(path, 'r')

    # Get data
    _states = dataset['train']['states'][:]
    _controls = dataset['train']['controls'][:]
    time = dataset['time'][:]

    # Normalize data
    states_normalizer = normalizer(_states)
    controls_normalizer = normalizer(_controls)

    states = states_normalizer.normalize(_states)
    controls = controls_normalizer.normalize(_controls)

    states_test = states_normalizer.normalize(dataset['test']['states'][:])
    controls_test = controls_normalizer.normalize(dataset['test']['controls'][:])

    # Define search space
    if type(cfg.dmd_training) == edmdc_training_class:
        rbf_cfg = cfg.dmd_training.observable
        bounds = [
            (cfg.dmd_training.observable.centers_lower_bound, cfg.dmd_training.observable.centers_upper_bound),
            (cfg.dmd_training.observable.centers_lower_bound, cfg.dmd_training.observable.centers_upper_bound),
            (cfg.dmd_training.observable.n_centers_lower_bound, cfg.dmd_training.observable.n_centers_upper_bound),
            (cfg.dmd_training.observable.kernel_width_lower_bound, cfg.dmd_training.observable.kernel_width_upper_bound),
            (0, 1)
        ]
        integrality = [
            False,
            False,
            True,
            False,
            True
        ]

        constraint = scipy.optimize.LinearConstraint([1, -1, 0, 0, 0],  -np.inf, -0.01, keep_feasible=True),

        # define callback to log value to mlflow
        def log_to_mlflow(intermediate_result: scipy.optimize.OptimizeResult):
            mlflow.log_metric('mse_best', intermediate_result.fun)
            rmse = np.sqrt(intermediate_result.fun)
            mlflow.log_metric('rmse_best', rmse)
            x = intermediate_result.x
            mlflow.log_metric('c1', x[0])
            mlflow.log_metric('c2', x[1])
            mlflow.log_metric('n_centers', x[2])
            mlflow.log_metric('kernel_width', x[3])
            mlflow.log_metric('include_states', x[4])

        # test one call
        logging.info('testing one call')
        fit_edmdc([bound[0] for bound in bounds], 
                cfg.dmd_training, states, controls, test=False)

        # Run optimization
        result = scipy.optimize.differential_evolution(
            fit_edmdc,
            bounds=bounds,
            args=(cfg.dmd_training, states, controls, False, states_test, controls_test),
            strategy='best1bin',
            maxiter=cfg.dmd_training.max_epochs,
            popsize=cfg.dmd_training.popsize,
            tol=cfg.dmd_training.tol,
            mutation=(0.5, 1),
            recombination=0.7,
            integrality=integrality,
            seed=42,
            updating='immediate',
            workers=cfg.n_workers if cfg.n_workers is not None else -1,
            constraints=(constraint),
            callback=None,
            disp=True,
            polish=False,
            init='latinhypercube',
            atol=cfg.dmd_training.tol,
        )

        logging.info('Optimization result: {}'.format(result))

        # Save best parameters
        # find best parameters in population
        results = result.population_energies
        result_idx = np.argmin(results)
        min_mse = results[result_idx]
        best_param = result.population[result_idx]
        min_rmse = np.sqrt(min_mse)
        mlflow.log_metric('mse_best', min_mse)
        mlflow.log_metric('rmse_best', min_rmse)

        c1, c2, n_centers, kernel_width, include_states = best_param
        n_centers = int(n_centers)
        include_states = bool(include_states)
        logging.info('Optimization result: mse = {} / rmse = {} for c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(min_mse, min_rmse, best_param[0], best_param[1], best_param[2], best_param[3], best_param[4]))
        if 'win' in sys.platform:
            print('Optimization result: mse = {} / rmse = {} for c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(min_mse, min_rmse, best_param[0], best_param[1], best_param[2], best_param[3], best_param[4]))
        # print also to file
        _path = dir_current_hydra_output() / 'best_parameters.json'
        with open(_path, 'w') as f:
            f.write('c1: {}, c2: {}, n_centers: {}, kernel_width: {}, include_states: {}'.format(c1, c2, n_centers, kernel_width, include_states))

    elif type(cfg.dmd_training) == dmdc_training_class:
        # loop through different ranks with dask multiprocessing
        results = []
        svd_ranks = range(2, max(states.shape[1], 40))
        for rank in svd_ranks:
            # results.append(dask.delayed(fit_edmdc)([rank], cfg.dmd_training, states, controls, test=False))
            results.append(fit_edmdc([rank], cfg.dmd_training, states, controls, test=False, states_test=states_test, controls_test=controls_test))
        # results = dask.compute(*results, num_workers=cfg.n_workers if cfg.n_workers is not None else -1,
                            #    scheduler='processes')
        results = np.array(results)
        result_idx = np.argmin(results)
        min_mse = results[result_idx]
        min_rank = svd_ranks[result_idx]
        logging.info('Optimization result: mse = {} / rmse = {} for rank = {}'.format(min_mse, np.sqrt(min_mse), min_rank))
        if 'win' in sys.platform:
            print('Optimization result: mse = {} / rmse = {} for rank = {}'.format(min_mse, np.sqrt(min_mse), min_rank))
        best_param = [min_rank]
        mlflow.log_metric('mse_best', min_mse)
        mlflow.log_metric('rmse_best', np.sqrt(min_mse))
        mlflow.log_metric('rank_best', min_rank)
        # save rank - rmse - mse as csv
        _path = dir_current_hydra_output() / 'rank_rmse_mse.csv'
        with open(_path, 'w') as f:
            f.write('rank,rmse,mse\n')
            for i in range(len(svd_ranks)):
                f.write('{},{},{}\n'.format(svd_ranks[i], np.sqrt(results[i]), results[i]))

    # # calculate mse on test set
    # mse = fit_edmdc(best_param, cfg.dmd_training, states_test, controls_test, test=True)

    # # log to mlflow
    # mlflow.log_metric('mse_test', mse)
    # mlflow.log_metric('rmse_test', np.sqrt(mse))

@hydra.main(config_path=str(Path('conf').absolute()), config_name='train_test_dmd', version_base=None)
def main(cfg: train_test_dmd_config_class):
    train_edmdc(cfg)

if __name__ == '__main__':
    main()