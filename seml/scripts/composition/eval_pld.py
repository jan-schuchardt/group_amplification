import numpy as np
import pickle
import seml
from sacred import Experiment
from tqdm import tqdm

from group_amplification.privacy_analysis.base_mechanisms import (GaussianMechanism,
                                                                  LaplaceMechanism,
                                                                  RandomizedResponseMechanism)

import group_amplification.privacy_analysis.composition.pld.accounting as pld_accounting

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
            )

    epsilons = {
        'space': 'linear_continuous',
        'params': {
            'start': 1,
            'stop': 8,
            'num': 8,
        }
    }

    base_mechanism = {
        'name': 'gaussian',
        'params': {
            'standard_deviation': 1.0
        }
    }

    amplification = {
        'subsampling_scheme': 'poisson',
        'tight': False,
        'params': {
            'subsampling_rate': 0.001,
            'group_size': 1,
            'insertions': 0,
            'iterations': 1000,
            'eval_params': {
                'pessimistic_estimate': True,
                'value_discretization_interval': 1e-4,
                'log_mass_truncation_bound': -50.0,
                'use_connect_dots': True
            }
        }
    }

    # Change this for camera ready
    # save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/icml24/results'
    save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/neurips24/composition/pld/poisson'


GaussianMechanism = ex.capture(GaussianMechanism, prefix='base_mechanism.params')
LaplaceMechanism = ex.capture(LaplaceMechanism, prefix='base_mechanism.params')
RandomizedResponseMechanism = ex.capture(RandomizedResponseMechanism, prefix='base_mechanism.params')
pld_traditional_group = ex.capture(pld_accounting.pld_traditional_group, prefix='amplification.params')
pld_tight_group = ex.capture(pld_accounting.pld_tight_group, prefix='amplification.params')


@ex.automain
def main(_config, epsilons, base_mechanism, amplification, save_dir):

    # epsilons
    continuous = epsilons['space'].lower().endswith('continuous')

    if epsilons['space'].lower().startswith('log'):
        epsilons = np.logspace(**epsilons['params'])
    elif epsilons['space'].lower().startswith('linear'):
        epsilons = np.linspace(**epsilons['params'])
    else:
        raise ValueError('epsilons.space should be in ["log", "linear"]')

    if not continuous:
        epsilons = np.sort(np.unique(epsilons.astype(int)))

    # Base mechanism
    if base_mechanism['name'].lower() == 'gaussian':
        base_mechanism = GaussianMechanism()
    elif base_mechanism['name'].lower() == 'laplace':
        base_mechanism = LaplaceMechanism()
    elif base_mechanism['name'].lower() == 'randomizedresponse':
        base_mechanism = RandomizedResponseMechanism()
    else:
        raise ValueError('base_mechanism.name should be in '
                         '["Gaussian", "RandomizedResponse"]')

    # Verify subsampling scheme parameter
    if not amplification['subsampling_scheme'].lower() == 'poisson':
        raise ValueError('amplification.subsampling scheme '
                         'should be "Poisson"')

    # Neighboring relation parameters
    group_size = amplification['params']['group_size']
    if group_size < 1:
        raise ValueError('Group size must be positive.')    
    insertions = amplification['params']['insertions']
    if insertions < 0:
        raise ValueError('Number of insertions must be non-negative.')
    if insertions > group_size:
        raise ValueError('Number of insertions must not exceed group size')
    deletions = group_size - insertions

    # Evaluate PLD guarantees
    if amplification['tight']:
        results = pld_tight_group(epsilons, base_mechanism,
                                  insertions=insertions, deletions=deletions)

    else:
        results = pld_traditional_group(epsilons, base_mechanism)

    # Save results
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    dict_to_save = {'epsilons': epsilons,
                    'results': results}

    save_file = f'{save_dir}/{db_collection}_{run_id}'

    with open(save_file, 'wb') as f:
        pickle.dump(dict_to_save, f)

    return {
        'save_file': save_file
    }
