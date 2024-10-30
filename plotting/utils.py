import os
import pandas as pd
from tqdm import tqdm
import numpy as np


def load_results(generate_result_fn, experiments=None, results_file=None, overwrite=False):
    if experiments is None and results_file is None:
        raise ValueError('Need to provide experiments or/and results_file')

    if experiments is None and not os.path.exists(results_file):
        raise ValueError('Results file does not exist')

    if overwrite or (results_file is None) or (results_file is not None and not os.path.exists(results_file)):
        if experiments is None:
            raise ValueError('Need to provide experiments')

        # Generate data
        result_dicts = []

        for exp in tqdm(experiments):
            result_dicts.append(generate_result_fn(exp))

        results = pd.DataFrame(result_dicts)

        if results_file is not None:
            directory, filename = os.path.split(results_file)
            if not os.path.exists(directory):
                os.mkdir(directory)

            pd.to_pickle(results, results_file)

    else:
        if not os.path.exists(results_file):
            raise ValueError('Results file does not exist')

        results = pd.read_pickle(results_file)

    return results


def merge_guarantees(xs_1, xs_2, ys_1, ys_2, merge_op, significant_digits=12):
    xs_1 = np.round(xs_1, decimals=significant_digits)
    xs_2 = np.round(xs_2, decimals=significant_digits)

    res_dict = {}

    for x, y in zip(xs_1, ys_1):
        if x in res_dict:
            res_dict[x] = merge_op(res_dict[x], y)
        else:
            res_dict[x] = y

    for x, y in zip(xs_2, ys_2):
        if x in res_dict:
            res_dict[x] = merge_op(res_dict[x], y)
        else:
            res_dict[x] = y

    new_xs, new_ys = [], []

    for k, v in res_dict.items():
        new_xs.append(k)
        new_ys.append(v)

    new_xs = np.array(new_xs)
    new_ys = np.array(new_ys)

    order = np.argsort(new_xs)

    return new_xs[order], new_ys[order]
