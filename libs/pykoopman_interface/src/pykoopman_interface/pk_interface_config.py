from pydantic.dataclasses import dataclass, Field
from dataclasses import asdict
from pydantic import ValidationInfo
from pydantic.functional_validators import field_validator
from hydra.core.config_store import ConfigStore
import hydra
import numpy as np
from omegaconf import MISSING, OmegaConf
from omegaconf import DictConfig
from typing import Dict, List, Optional, Union
from pathlib import Path
from pydantic import model_validator
import logging
import yaml

"""dmd config dataclass definitions"""

@dataclass
class observable_class:
    pass

@dataclass
class rbf_observable_class(observable_class):
    rbf_type='thinplate'
    centers_lower_bound: float = -2.0
    centers_upper_bound: float = 2.0
    n_centers_lower_bound: int = 3
    n_centers_upper_bound: int = 200
    kernel_width_lower_bound: float = 0.001
    kernel_width_upper_bound: float = 10.0

@dataclass
class dmd_training_class:
    """this is the base class, in case other dmd methods are added"""
    n_samples_train: Optional[int] = None
    pass

@dataclass
class edmdc_training_class(dmd_training_class):
    n_samples_train: int = 256
    observable: observable_class = rbf_observable_class()
    max_epochs: int = 60
    popsize: int = 4
    tol: float = 1e-4
    pass

@dataclass
class dmdc_training_class(dmd_training_class):
    max_rank: Optional[int] = None
    min_rank: int = 1
    ignore_parameters: bool = False
    ignore_outputs: bool = False
    pass


'''train edmdc config dataclass definition'''

@dataclass
class train_test_dmd_config_class:
    # Provide dataset by path (preferred) or by name (legacy); only one should be set
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dmd_training: dmd_training_class = MISSING

    mlflow_experiment_name: str = 'Default'
    mlflow_tracking_uri: str = ''

    n_workers: Optional[int] = None # None is all cpus available
    raise_exception: bool = True
    
    @field_validator('dataset_name')
    @classmethod
    def set_dataset_name_from_path(cls, v, info: ValidationInfo):
        """Extract filename from dataset_path and set as dataset_name if not already set."""
        dataset_path = info.data.get('dataset_path')
        if dataset_path is not None:
            path = Path(dataset_path)
            # Extract filename without extension as dataset_name
            return path.stem
        return v




def get_config_store():
    cs = ConfigStore.instance()
    # dmd
    cs.store(name='base_train_test_dmd', node=train_test_dmd_config_class)
    cs.store(group='dmd_training', name='base_edmdc_training', node=edmdc_training_class())
    cs.store(group='dmd_training', name='base_dmdc_training', node=dmdc_training_class())
    return cs