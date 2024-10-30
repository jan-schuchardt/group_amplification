import numpy as np
import pickle
import seml
from sacred import Experiment
from tqdm import tqdm

from group_amplification.privacy_analysis.base_mechanisms import (GaussianMechanism,
                                                                  LaplaceMechanism,
                                                                  RandomizedResponseMechanism)
import group_amplification.privacy_analysis.subsampling.rdp.without_replacement as without_replacement
import group_amplification.privacy_analysis.composition.rdp.permutation as permutation


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
        'space': 'log_continuous',
        'params': {
            'start': 0,
            'stop': 2,
            'num': 21,
        }
    }

    base_mechanism = {
        'name': 'gaussian',
        'params': {
            'standard_deviation': 1.0
        }
    }

    amplification = {
        'subsampling_scheme': 'permutation',
        'params': {
            'n_chunks': 2,
            'n_iterations': 2,
            'eval_method': 'directtransport',
            'eval_params': {
                'dps': 15
            }
        }
    }

    # Change this for camera ready
    # save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/icml24/results'
    save_dir = '/ceph/hdd/staff/schuchaj/group_amplification_results/icml24_rebuttal/composition'


GaussianMechanism = ex.capture(GaussianMechanism, prefix='base_mechanism.params')
LaplaceMechanism = ex.capture(GaussianMechanism, prefix='base_mechanism.params')
RandomizedResponseMechanism = ex.capture(RandomizedResponseMechanism, prefix='base_mechanism.params')

without_replacement._rdp_self_consistency_quadrature = ex.capture(without_replacement._rdp_self_consistency_quadrature, prefix='amplification.params')
permutation.rdp_direct_transport = ex.capture(permutation.rdp_direct_transport, prefix='amplification.params')
permutation.rdp_tight = ex.capture(permutation.rdp_tight, prefix='amplification.params')


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
    valid_schemes = ['permutation', 'withoutreplacement']

    subsampling_scheme = amplification['subsampling_scheme'].lower()
    if subsampling_scheme not in valid_schemes:
        raise ValueError('amplification.subsampling scheme '
                         f'must be in "{valid_schemes}"')

    n_iterations = amplification['params']['n_iterations']
    n_chunks = amplification['params']['n_chunks']

    epsilons = []

    for alpha in tqdm(alphas):    
        if subsampling_scheme == 'withoutreplacement':
            epsilons.append(
                n_iterations * without_replacement._rdp_self_consistency_quadrature(
                                        alpha, base_mechanism,
                                        amplification['params']['n_iterations'],
                                        1))

        elif subsampling_scheme == 'permutation':
            eval_method = amplification['params'].get('eval_method', '').lower()

            if eval_method == 'directtransport':
                epsilons.append(
                    permutation.rdp_direct_transport(
                                        alpha, base_mechanism,
                                        n_chunks,
                                        n_iterations))

            elif eval_method == 'tight':
                epsilons.append(
                    permutation.rdp_tight(
                                        alpha, base_mechanism,
                                        n_chunks,
                                        n_iterations))

            else:
                raise ValueError('eval_method for permutation must be in '
                                 '["directtransport", "tight"]')

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
