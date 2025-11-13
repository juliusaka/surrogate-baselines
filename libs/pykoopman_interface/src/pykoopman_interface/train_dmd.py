import pykoopman as pk
import numpy as np
np.random.seed(42) # for reproducibility
import h5py
import hydra
import mlflow
import logging
import pandas as pd
# import dask
import sys
import json
import torch
from pathlib import Path
import shutil

from pykoopman_interface.pk_interface_config import get_config_store, train_test_dmd_config_class

# This is hacky as pykoopman is not really well packaged.
_path = str(Path(__file__).parent / './../../dependencies/bnode-core/src/bnode_core/nn/nn_utils')
print('Inserting path for bnode-core nn_utils:', _path)
print(f"path exists: {Path(_path).exists()}")
sys.path.insert(0, _path)

_path = str(Path(__file__).parent / './../../dependencies/bnode-core/src/bnode_core/utils/')
print('Inserting path for bnode-core utils:', _path)
print(f"path exists: {Path(_path).exists()}")
sys.path.insert(0, _path)


import warnings
import sys
from normalization import NormalizationLayer1D
from hydra_mlflow_decorator import log_hydra_to_mlflow

warnings.filterwarnings("ignore")

class normalization_model_class(torch.nn.Module):
    """
    A placeholder for the normalization model.
    This should be replaced with the actual normalization model implementation.
    """
    def __init__(self, state_dim: int, parameters_dim: int, controls_dim: int, output_dim: int):
        super(normalization_model_class, self).__init__()
        self.state_dim = state_dim
        self.parameters_dim = parameters_dim if parameters_dim > 0 else None
        self.controls_dim = controls_dim if controls_dim > 0 else None
        self.output_dim = output_dim if output_dim > 0 else None

        self.state_normalization = NormalizationLayer1D(self.state_dim)
        self.parameters_normalization = NormalizationLayer1D(self.parameters_dim) if self.parameters_dim is not None else None
        self.controls_normalization = NormalizationLayer1D(self.controls_dim) if self.controls_dim is not None else None
        self.output_normalization = NormalizationLayer1D(self.output_dim) if self.output_dim is not None else None

    def forward(self, states, controls=None, parameters = None, outputs=None, states_der=None, denormalize=False):
        """
        Normalize or denormalize the states, controls, and outputs.
        
        Args:
            states (torch.Tensor): The states to normalize/denormalize.
            controls (torch.Tensor, optional): The controls to normalize/denormalize.
            outputs (torch.Tensor, optional): The outputs to normalize/denormalize.
            denormalize (bool): If True, denormalizes the inputs; otherwise normalizes them.
        
        Returns:
            tuple: Normalized or denormalized states, controls, and outputs.
        """
        states = self.state_normalization(states, denormalize=denormalize)
        if self.parameters_normalization is not None and parameters is not None:
            parameters = self.parameters_normalization(parameters, denormalize=denormalize)
        else:
            parameters = None

        if self.controls_normalization is not None and controls is not None:
            controls = self.controls_normalization(controls, denormalize=denormalize)
        else:
            controls = None

        if self.output_normalization is not None and outputs is not None:
            outputs = self.output_normalization(outputs, denormalize=denormalize)
        else:
            outputs = None

        return {'states': states,
                'controls': controls,
                'parameters': parameters,
                'outputs': outputs,}
    
    def initialize_normalization(self, states, parameters = None, controls=None, outputs=None):
        """
        Initialize the normalization parameters for states, controls, and outputs.
        
        Args:
            states (torch.Tensor): The states to initialize normalization.
            controls (torch.Tensor, optional): The controls to initialize normalization.
            outputs (torch.Tensor, optional): The outputs to initialize normalization.
        """
        def reshape_array(array):
            arr = array.transpose(1,0,2).reshape(array.shape[1],-1).transpose(1,0)
            return arr
        self.state_normalization.initialize_normalization(reshape_array(states), name = 'states')
        if self.parameters_normalization is not None and parameters is not None:
            self.parameters_normalization.initialize_normalization(parameters, name = 'parameters')
        if self.controls_normalization is not None and controls is not None:
            self.controls_normalization.initialize_normalization(reshape_array(controls), name = 'controls')
        if self.output_normalization is not None and outputs is not None:
            self.output_normalization.initialize_normalization(reshape_array(outputs), name = 'outputs')
        


def train_edmdc(cfg, normalization_model, train_data, validation_data):
    """
    Train EDMDc model with genetic algorithm optimization.
    """
    model = None
    model_params = None
    return model, model_params

def aggregate_variables_dmdc(dict):
    _parameters = dict['parameters'] if 'parameters' in dict.keys() else None
    _states = dict['states']
    _controls = dict['controls'] if 'controls' in dict.keys() else None
    _outputs = dict['outputs'] if 'outputs' in dict.keys() else None
    _parameters = _parameters.unsqueeze(2).expand(-1,-1,_states.shape[2]) if _parameters is not None else None
    states = torch.cat([_states, _outputs], dim=1) if _outputs is not None else _states
    controls = torch.cat([_controls, _parameters], dim=1) if _controls is not None and _parameters is not None else _controls
    return {'states': states, 'controls': controls}

def reverse_aggregate_variables_dmdc(states_dmdc, state_dim):
    """
    Reverse the aggreation of states and outputs
    """
    states = states_dmdc[:,:state_dim,:]
    if state_dim < states_dmdc.shape[1]:
        outputs = states_dmdc[:,state_dim:,:]
    else:
        outputs = None
    return states, outputs

def convert_hdf5_to_torch_tensor_dict(data):
    """
    Convert HDF5 data to torch tensors.
    
    Args:
        data (dict): A dictionary containing HDF5 data.
    
    Returns:
        dict: A dictionary with torch tensors.
    """
    converted_data = {}
    for key, value in data.items():
        converted_data[key] = torch.tensor(value, dtype=torch.float32)
    return converted_data

def convert_torch_tensor_to_numpy(data):
    """
    Convert torch tensors to numpy arrays.
    
    Args:
        data (dict): A dictionary containing torch tensors.
    
    Returns:
        dict: A dictionary with numpy arrays.
    """
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            converted_data[key] = value.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported data type for key {key}: {type(value)}")
    return converted_data

def calculate_dmdc_mse(model, dmdc_data):
    dmdc_states = dmdc_data['states']
    dmdc_controls = dmdc_data['controls']
    mse_list = []
    for i in range(dmdc_states.shape[0]):
        x = dmdc_states[i]
        u = dmdc_controls[i]
        x0 = x[:,0] # x is now of shape (number_of_channels, sequence_length)
        
        xhat = model.simulate(x0, u.T, n_steps=x.shape[1]-1) # u.T to get shape (sequence_length, number_of_channels)
        xhat = xhat.T # transpose to get shape (sequence_length, number_of_channels)
        mse = np.mean(np.square(xhat - x[:,1:]))
        mse_list.append(mse)
    mse = np.mean(mse_list)
    std = np.std(mse_list)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'std': std, 'rmse': rmse}

def generate_predictions_dmdc(model, normalization_model, data):
    dmdc_data = prepare_data_for_dmdc(normalization_model, data)
    dmdc_states = dmdc_data['states']
    dmdc_controls = dmdc_data['controls']
    dmdc_predictions = []
    for i in range(dmdc_states.shape[0]):
        x = dmdc_states[i]
        u = dmdc_controls[i]
        x0 = x[:,0]
        xhat = model.simulate(x0, u.T, n_steps=x.shape[1]-1)
        xhat = xhat.T # transpose to get shape (sequence_length, number_of_channels)
        # add x0 to the beginning of xhat
        xhat = np.concatenate([np.expand_dims(x0,1), xhat], axis=1)
        dmdc_predictions.append(xhat)
    dmdc_predictions = np.array(dmdc_predictions)
    states_norm, outputs_norm = reverse_aggregate_variables_dmdc(dmdc_predictions, data['states'].shape[1])
    states = normalization_model.state_normalization(torch.tensor(states_norm, dtype=torch.float32), denormalize=True).numpy()
    outputs = normalization_model.output_normalization(torch.tensor(outputs_norm, dtype=torch.float32), denormalize=True).numpy() if outputs_norm is not None else None
    predictions = {'states': states, 'outputs': outputs}
    return predictions

def fit_dmdc(train_data, rank: int):
    dmdc_regressor = pk.regression.DMDc(
        svd_rank=rank + train_data['controls'].shape[1],
        svd_output_rank = rank
    )
    model = pk.Koopman(regressor=dmdc_regressor)
    x = train_data['states'][:,:,:-1]
    xp1 = train_data['states'][:,:,1:]
    u = train_data['controls'][:,:,:-1]
    # concatenate timesteps
    X = np.zeros((x.shape[1], x.shape[0] * x.shape[2]))
    Xp1 = np.zeros((x.shape[1], x.shape[0] * x.shape[2]))
    U = np.zeros((u.shape[1], u.shape[0] * u.shape[2]))
    for i in range(x.shape[0]):
        X[:, i * x.shape[2]:(i + 1) * x.shape[2]] = x[i]
        Xp1[:, i * x.shape[2]:(i + 1) * x.shape[2]] = xp1[i]
        U[:, i * u.shape[2]:(i + 1) * u.shape[2]] = u[i]
    # fit the model
    model.fit(X.T, Xp1.T, U.T) # transpose to get shape (n_samples, n_features)
    return model

def append_context_key(_dict: dict, context: str):
    dict = {}
    for key, value in _dict.items():
        dict[f'{key}_{context}'] = value
    return dict

def fit_dmdc_until_max_rank(cfg, normalization_model, _train_data, _validation_data, test_data):
    # prepare data by normalizing it and aggregating the variables (e.g. states + controls, states + outputs)
    dmdc_train_data_norm = prepare_data_for_dmdc(normalization_model, _train_data)
    dmdc_validation_data_norm = prepare_data_for_dmdc(normalization_model, _validation_data)
    dmdc_test_data_norm = prepare_data_for_dmdc(normalization_model, test_data)

    if cfg.dmd_training.n_samples_train is not None:
        logging.info('Using {} samples for dmdc training'.format(cfg.dmd_training.n_samples_train))
        idx = np.random.choice(dmdc_train_data_norm['states'].shape[0], cfg.dmd_training.n_samples_train)
        for key in dmdc_train_data_norm.keys():
            dmdc_train_data_norm[key] = dmdc_train_data_norm[key][idx]

    param_dim = 0 if normalization_model.parameters_dim is None else normalization_model.parameters_dim
    controls_dim = 0 if normalization_model.controls_dim is None else normalization_model.controls_dim
    output_dim = 0 if normalization_model.output_dim is None else normalization_model.output_dim
    max_possible_rank = normalization_model.state_dim + param_dim + controls_dim + output_dim
    if cfg.dmd_training.max_rank is None:
        max_rank = max_possible_rank
    else:
        max_rank = min(cfg.dmd_training.max_rank, max_possible_rank)
        if cfg.dmd_training.max_rank > max_possible_rank:
            logging.warning('Max rank {} is larger than the maximum possible rank {}. Using max possible rank.'.format(max_rank, max_possible_rank))
        logging.info('Using max rank of {} for DMDc from config'.format(max_rank))
    logging.info('Training of {} DMDc models with ranks from {} to {}'.format(max_rank - cfg.dmd_training.min_rank, cfg.dmd_training.min_rank, max_rank))

    training_results = []
    models = []
    for svd_rank in range(cfg.dmd_training.min_rank, max_rank + 1):
        model = fit_dmdc(dmdc_train_data_norm, svd_rank)
        res_train = calculate_dmdc_mse(model, dmdc_train_data_norm)
        res_val = calculate_dmdc_mse(model, dmdc_validation_data_norm)
        res_test = calculate_dmdc_mse(model, dmdc_test_data_norm)
        training_results.append({**append_context_key(res_train, 'train'),
                                 **append_context_key(res_val, 'validation'),
                                 **append_context_key(res_test, 'test'),
                                 'rank': svd_rank})
        models.append(model)
        mlflow.log_metrics(append_context_key(res_train, 'train'), step=svd_rank)
        mlflow.log_metrics(append_context_key(res_val, 'validation'), step=svd_rank)
        mlflow.log_metrics(append_context_key(res_test, 'test'), step=svd_rank)
        logging.info('Rank: {}, Train RMSE: {:4f} +- {:4f}, Validation RMSE: {:4f} +- {:4f}, Test RMSE: {:4f} +- {:4f}'.format(
            svd_rank,
            res_train['rmse'], res_train['std'],
            res_val['rmse'], res_val['std'],
            res_test['rmse'], res_test['std']
        ))
    return models, training_results

def prepare_data_for_dmdc(normalization_model, _data):
    _data_norm = normalization_model(**convert_hdf5_to_torch_tensor_dict(_data))
    dmdc_data_norm = convert_torch_tensor_to_numpy(aggregate_variables_dmdc(_data_norm))
    return dmdc_data_norm

@log_hydra_to_mlflow
def train_for_best_model(cfg: train_test_dmd_config_class):
    """
    This function does:
    - Loading and normalizing the data
    - train 
    """

    logging.info('starting training')
    logging.info('train test dmd config: {}'.format(cfg))

    # Load data
    path = filepath_dataset_from_name(cfg.dataset_name)
    dataset = h5py.File(path, 'r')

    # Get data
    train_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']

    # delete entries from dataset for testing
    # create torch model for normalization
    ignore_parameters = cfg.dmd_training.ignore_parameters
    ignore_outputs = cfg.dmd_training.ignore_outputs
    _states = train_data['states'][:]
    _parameters = train_data['parameters'][:] if 'parameters' in train_data.keys() and ignore_parameters is False else None
    _controls = train_data['controls'][:] if 'controls' in train_data.keys() else None
    _outputs = train_data['outputs'][:] if 'outputs' in train_data.keys() and ignore_outputs is False else None

    normalization_model = normalization_model_class(
        state_dim=_states.shape[1],
        parameters_dim=_parameters.shape[1] if _parameters is not None else 0,
        controls_dim=_controls.shape[1] if _controls is not None else 0,
        output_dim=_outputs.shape[1] if _outputs is not None else 0
    )
    normalization_model.initialize_normalization(
        states=_states,
        parameters=_parameters,
        controls=_controls,
        outputs=_outputs
    )
    logging.info('Normalization model initialized with state_dim: {}, parameters_dim: {}, controls_dim: {}, output_dim: {}'.format(
        normalization_model.state_dim, normalization_model.parameters_dim, normalization_model.controls_dim, normalization_model.output_dim))

    # Train edmdc
    if type(cfg.dmd_training) == edmdc_training_class:
        logging.info('Training EDMDc model')
        models, training_results = train_edmdc(cfg, normalization_model, train_data, validation_data, test_data)


    # fit dmdc
    elif type(cfg.dmd_training) == dmdc_training_class:
        logging.info('Fitting DMDc models for all ranks')
        models, training_results = fit_dmdc_until_max_rank(cfg, normalization_model, train_data, validation_data, test_data)
        pd.DataFrame(training_results).to_csv(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / 'dmdc_training_results.csv', index=False)
        
        # test the best model
        best_idx = np.argmin([res['rmse_validation'] for res in training_results])
        best_rank = training_results[best_idx]['rank']
        logging.info('Best model found with rank {} and train/test/val RMSE: {:.4f}/{:.4f}/{:.4f}'.format(
            best_rank,
            training_results[best_idx]['rmse_train'],
            training_results[best_idx]['rmse_test'],
            training_results[best_idx]['rmse_validation']
        ))
        best_model = models[best_idx]
        # save the best model settings
        best_model_params = {
            'rank': best_rank,
        }        
        # save as json to hydra output directory
        with open(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / 'best_model_params.json', 'w') as f:
            json.dump(best_model_params, f)
        # copy dataset to hydra output directory and save predictions of the best model
        dataset_copy_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / 'dataset.hdf5'
        shutil.copy(path, dataset_copy_path)
        test_dataset = h5py.File(dataset_copy_path, 'r+')
        # generate predictions for the best model
        for context, data in {'train': train_data, 'validation': validation_data, 'test': test_data}.items():
            logging.info('Generating predictions for {} dataset'.format(data))
            predictions = generate_predictions_dmdc(best_model, normalization_model, data)
            # save predictions to dataset
            for key, value in predictions.items():
                if value is not None:
                    test_dataset.create_dataset(f'{context}/{key}_hat', data=value)
                    rmse = np.sqrt(np.mean(np.square(value - data[key][:])))
                    logging.info('unnormalized RMSE for {} dataset and key {}: {:.4f}'.format(context, key, rmse))
        test_dataset.close()
        logging.info('Saved best model predictions to dataset at {}'.format(dataset_copy_path))
    
@hydra.main(config_path=str(Path('conf').absolute()), config_name='train_test_dmd', version_base=None)
def main(cfg: train_test_dmd_config_class):
    train_for_best_model(cfg)

if __name__ == '__main__':
    main()