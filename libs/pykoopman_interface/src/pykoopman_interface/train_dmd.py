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

from pykoopman_interface.log_hydra_to_mlflow import log_hydra_to_mlflow  

# This is hacky as pykoopman is not really well packaged.
_path = str(Path(__file__).parent / './../../dependencies/bnode-core/src/bnode_core/nn/nn_utils')
print('Inserting path for bnode-core nn_utils:', _path)
print(f"path exists: {Path(_path).exists()}")
sys.path.insert(0, _path)

import warnings
from normalization import NormalizationLayer1D  # type: ignore

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
    """
    Calculate MSE and RMSE on normalized data for validation.
    No variance calculation.
    """
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
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def calculate_detailed_metrics(model, normalization_model, data, state_dim):
    """
    Calculate detailed metrics for states and outputs separately on normalized data.
    
    Returns a dict with keys:
    - mse_states, rmse_states
    - mse_outputs, rmse_outputs (if outputs exist)
    - mse_states_0, rmse_states_0 (initial state error)
    - mse_outputs_0, rmse_outputs_0 (initial output error, if outputs exist)
    
    All calculations are performed on normalized data.
    """
    dmdc_data = prepare_data_for_dmdc(normalization_model, data)
    dmdc_states = dmdc_data['states']
    dmdc_controls = dmdc_data['controls']
    
    # Get predictions in normalized space
    dmdc_predictions = []
    for i in range(dmdc_states.shape[0]):
        x = dmdc_states[i]
        u = dmdc_controls[i]
        x0 = x[:,0]
        xhat = model.simulate(x0, u.T, n_steps=x.shape[1]-1)
        xhat = xhat.T
        # add x0 to the beginning of xhat
        xhat = np.concatenate([np.expand_dims(x0,1), xhat], axis=1)
        dmdc_predictions.append(xhat)
    dmdc_predictions = np.array(dmdc_predictions)
    
    # Separate states and outputs in normalized space
    states_norm_pred, outputs_norm_pred = reverse_aggregate_variables_dmdc(dmdc_predictions, state_dim)
    states_norm_true, outputs_norm_true = reverse_aggregate_variables_dmdc(dmdc_states, state_dim)
    
    metrics = {}
    
    # States metrics (all timesteps) - on normalized data
    mse_states = np.mean(np.square(states_norm_pred - states_norm_true))
    metrics['mse_states'] = mse_states
    metrics['rmse_states'] = np.sqrt(mse_states)
    
    # States initial condition (t=0) - on normalized data
    mse_states_0 = np.mean(np.square(states_norm_pred[:, :, 0] - states_norm_true[:, :, 0]))
    metrics['mse_states_0'] = mse_states_0
    metrics['rmse_states_0'] = np.sqrt(mse_states_0)
    
    # Outputs metrics (if outputs exist) - on normalized data
    if outputs_norm_pred is not None and normalization_model.output_dim is not None:
        # Outputs metrics (all timesteps) - on normalized data
        mse_outputs = np.mean(np.square(outputs_norm_pred - outputs_norm_true))
        metrics['mse_outputs'] = mse_outputs
        metrics['rmse_outputs'] = np.sqrt(mse_outputs)
        
        # Outputs initial condition (t=0) - on normalized data
        mse_outputs_0 = np.mean(np.square(outputs_norm_pred[:, :, 0] - outputs_norm_true[:, :, 0]))
        metrics['mse_outputs_0'] = mse_outputs_0
        metrics['rmse_outputs_0'] = np.sqrt(mse_outputs_0)
    else:
        metrics['mse_outputs'] = None
        metrics['rmse_outputs'] = None
        metrics['mse_outputs_0'] = None
        metrics['rmse_outputs_0'] = None
    
    return metrics

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
    dmdc_validation_data_norm = prepare_data_for_dmdc(normalization_model, _validation_data) # TODO: why not used?
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
    state_dim = normalization_model.state_dim
    
    for svd_rank in range(cfg.dmd_training.min_rank, max_rank + 1):
        model = fit_dmdc(dmdc_train_data_norm, svd_rank)
        
        # Calculate detailed metrics for train, validation, and test
        metrics_train = calculate_detailed_metrics(model, normalization_model, _train_data, state_dim)
        metrics_validation = calculate_detailed_metrics(model, normalization_model, _validation_data, state_dim)
        metrics_test = calculate_detailed_metrics(model, normalization_model, test_data, state_dim)
        
        # Build result dict with new naming convention
        result = {'rank': svd_rank}
        
        # Add train metrics #TODO: could do this for all contexts in a loop
        for key, value in metrics_train.items():
            if value is not None:
                result[f'{key}_train'] = value
        
        # Add validation metrics
        for key, value in metrics_validation.items():
            if value is not None:
                result[f'{key}_validation'] = value
        
        # Add test metrics
        for key, value in metrics_test.items():
            if value is not None:
                result[f'{key}_test'] = value
        
        training_results.append(result)
        models.append(model)
        
        # Log all metrics to MLflow (with step for tracking across ranks)
        mlflow_metrics = {k: v for k, v in result.items() if k != 'rank' and v is not None}
        mlflow.log_metrics(mlflow_metrics, step=svd_rank)
        
        # Enhanced logging output
        log_msg = f'Rank: {svd_rank}'
        log_msg += f'\n  Train  - States RMSE: {metrics_train["rmse_states"]:.6f}'
        if metrics_train['rmse_outputs'] is not None:
            log_msg += f', Outputs RMSE: {metrics_train["rmse_outputs"]:.6f}'
        log_msg += f'\n  Val    - States RMSE: {metrics_validation["rmse_states"]:.6f}'
        if metrics_validation['rmse_outputs'] is not None:
            log_msg += f', Outputs RMSE: {metrics_validation["rmse_outputs"]:.6f}'
        log_msg += f'\n  Test   - States RMSE: {metrics_test["rmse_states"]:.6f}'
        if metrics_test['rmse_outputs'] is not None:
            log_msg += f', Outputs RMSE: {metrics_test["rmse_outputs"]:.6f}'
        logging.info(log_msg)
        
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
    path = Path(cfg.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path {path} does not exist.")
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

    # Fit DMDc models only (EDMDc training removed)
    logging.info('Fitting DMDc models for all ranks')
    models, training_results = fit_dmdc_until_max_rank(cfg, normalization_model, train_data, validation_data, test_data)
    pd.DataFrame(training_results).to_csv(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / 'dmdc_training_results.csv', index=False)
    
    # Select best model based on sum of validation states RMSE and (if available) outputs RMSE.
    # Requirement: discard NaNs by replacing with inf; if outputs RMSE not present, use only states RMSE.
    def _selection_score(res):
        states_rmse = res.get('rmse_states_validation')
        outputs_rmse = res.get('rmse_outputs_validation')
        # Replace NaNs with inf for states; ignore missing/NaN outputs by treating as 0 in sum.
        if states_rmse is None or (isinstance(states_rmse, float) and np.isnan(states_rmse)):
            states_rmse = np.inf
        if outputs_rmse is None or (isinstance(outputs_rmse, float) and np.isnan(outputs_rmse)):
            outputs_rmse = 0.0  # Not available or NaN: don't penalize selection beyond states
        return states_rmse + outputs_rmse
    _selection_scores = [_selection_score(res) for res in training_results]
    best_idx = int(np.argmin(_selection_scores))
    logging.info(f'Model selection scores (states + outputs RMSE): {_selection_scores}')
    best_rank = training_results[best_idx]['rank']
    
    # Log final metrics for the best model
    final_metrics = {}
    for key, value in training_results[best_idx].items():
        if key != 'rank' and value is not None:
            # Add _final suffix to all train/validation/test metrics
            final_metrics[f'{key}_final'] = value
    
    mlflow.log_metrics(final_metrics)
    
    # Enhanced logging for best model
    logging.info(f'Best model found with rank {best_rank}')
    logging.info(f'  Train - States RMSE: {training_results[best_idx]["rmse_states_train"]:.6f}, MSE: {training_results[best_idx]["mse_states_train"]:.6f}')
    if training_results[best_idx].get('rmse_outputs_train') is not None:
        logging.info(f'  Train - Outputs RMSE: {training_results[best_idx]["rmse_outputs_train"]:.6f}, MSE: {training_results[best_idx]["mse_outputs_train"]:.6f}')
    logging.info(f'  Val   - States RMSE: {training_results[best_idx]["rmse_states_validation"]:.6f}, MSE: {training_results[best_idx]["mse_states_validation"]:.6f}')
    if training_results[best_idx].get('rmse_outputs_validation') is not None:
        logging.info(f'  Val   - Outputs RMSE: {training_results[best_idx]["rmse_outputs_validation"]:.6f}, MSE: {training_results[best_idx]["mse_outputs_validation"]:.6f}')
    logging.info(f'  Test  - States RMSE: {training_results[best_idx]["rmse_states_test"]:.6f}, MSE: {training_results[best_idx]["mse_states_test"]:.6f}')
    if training_results[best_idx].get('rmse_outputs_test') is not None:
        logging.info(f'  Test  - Outputs RMSE: {training_results[best_idx]["rmse_outputs_test"]:.6f}, MSE: {training_results[best_idx]["mse_outputs_test"]:.6f}')
    
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
    

def main():
    cs = get_config_store()
    hydra.main(config_path=str(Path('config').absolute()), config_name='train_test_dmd', version_base=None)(train_for_best_model)()

if __name__ == '__main__':
    main()