import os

from models.coop import coop
from models.csp import get_csp, get_mix_csp
from models.proda import get_proda
from models.condcsp import get_condcsp
from models.dfsp import get_dfsp
from models.hypercsp import get_hypercsp
from models.gencsp import get_gencsp


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device, is_training=True):
    if config.experiment_name == "coop":
        return coop(train_dataset, config, device, is_training)

    elif config.experiment_name == "csp":
        return get_csp(train_dataset, config, device, is_training)

    # special experimental setup
    elif config.experiment_name == "mix_csp":
        return get_mix_csp(train_dataset, config, device, is_training)
    
    elif config.experiment_name == 'proda':
        return get_proda(train_dataset, config, device, is_training)
    
    elif config.experiment_name == 'condcsp':
        return get_condcsp(train_dataset, config, device, is_training)
    
    elif config.experiment_name == 'dfsp':
        return get_dfsp(train_dataset, config, device, is_training)

    elif config.experiment_name == 'hypercsp':
        return get_hypercsp(train_dataset, config, device, is_training)
    
    elif config.experiment_name == 'gencsp':
        return get_gencsp(train_dataset, config, device, is_training)
    
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )
