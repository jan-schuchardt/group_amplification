import numpy as np
import pickle
import seml
from sacred import Experiment
from tqdm import tqdm

from group_amplification.privacy_analysis.base_mechanisms import (GaussianMechanism,
                                                                  LaplaceMechanism,
                                                                  RandomizedResponseMechanism)
import group_amplification.privacy_analysis.subsampling.rdp.poisson as poisson

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

    alphas = {
        'space': 'log',
        'params': {
            'start': 0,
            'stop': 3,
            'num': 91,
        }
    }

    base_mechanism = {
        'name': 'gaussian',
        'params': {
            'standard_deviation': 5.0
        }
    }

    amplification = {
        'subsampling_scheme': 'poisson',
        'tight': False,
        'params': {
            'subsampling_rate': 0.001,
            'group_size': 1,
            'insertions': 0,
            'eval_method': 'recursive',
            'eval_params': {
                'hoelders_exponent': 2,
                'dps': 50
            }
        }
    }

    # Change this for camera ready
    # save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/icml24/results'
    save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/icml24/poisson'


GaussianMechanism = ex.capture(GaussianMechanism, prefix='base_mechanism.params')
LaplaceMechanism = ex.capture(LaplaceMechanism, prefix='base_mechanism.params')
RandomizedResponseMechanism = ex.capture(RandomizedResponseMechanism, prefix='base_mechanism.params')
poisson.rdp_traditional_group = ex.capture(poisson.rdp_traditional_group, prefix='amplification.params')
poisson.rdp_tight_group = ex.capture(poisson.rdp_tight_group, prefix='amplification.params')


@ex.automain
def main(_config, alphas, base_mechanism, amplification, save_dir):

    # Alphas
    continuous = alphas['space'].lower().endswith('continuous')

    if alphas['space'].lower().startswith('log'):
        alphas = np.logspace(**alphas['params'])
    elif alphas['space'].lower().startswith('linear'):
        alphas = np.linspace(**alphas['params'])
    else:
        raise ValueError('alphas.space should be in ["log", "linear"]')

    if not continuous:
        alphas = np.sort(np.unique(alphas.astype(int)))
    alphas = alphas[alphas > 1]

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
        raise ValueError('Number of insertions must not exceed gruop size')
    deletions = group_size - insertions

    # Evaluate RDP guarantees
    if amplification['tight']:
        epsilons = np.array(
            [poisson.rdp_tight_group(alpha, base_mechanism,
                                     insertions=insertions, deletions=deletions)
             for alpha in tqdm(alphas)])

    else:
        epsilons = np.array(
            [poisson.rdp_traditional_group(alpha, base_mechanism) for alpha in tqdm(alphas)]
        )

    # Save results
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    dict_to_save = {'alphas': alphas,
                    'epsilons': epsilons,
                    'config': _config}

    save_file = f'{save_dir}/{db_collection}_{run_id}'

    with open(save_file, 'wb') as f:
        pickle.dump(dict_to_save, f)

    return {
        'save_file': save_file
    }