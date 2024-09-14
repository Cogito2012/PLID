import os

from models.csp import get_csp
from models.gencsp import get_gencsp


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device, is_training=True):
    if config.experiment_name == "csp":
        return get_csp(train_dataset, config, device, is_training)
    
    elif config.experiment_name == 'gencsp':
        return get_gencsp(train_dataset, config, device, is_training)
    
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )
