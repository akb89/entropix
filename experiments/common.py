"""Common functions for XP."""
import os

from collections import defaultdict
from tqdm import tqdm

import numpy as np

import entropix.utils.data as dutils
import entropix.utils.files as futils
import entropix.utils.metrix as metrix
import entropix.core.aligner as aligner
import entropix.core.matrixor as matrixor
import entropix.core.evaluator as evaluator

__all__ = ('launch_xp', 'dump_aligned_models', 'load_aligned_models')


def dump_aligned_models(models, dirpath):
    print('Dumping aligned models and vocabularies...')
    for name1, model1, vocab1 in tqdm(models):
        aligned_model1 = model1
        vocab = vocab1
        for name2, model2, vocab2 in models:
            if name1 == name2:
                continue
            assert aligned_model1.shape[1] == model2.shape[1]
            aligned_model1, _, vocab = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
        vocab_path = os.path.join(dirpath, '{}-aligned.vocab'.format(name1))
        model_path = os.path.join(dirpath, '{}-aligned.npy'.format(name1))
        futils.save_vocab(vocab, vocab_path)
        np.save(model_path, aligned_model1)


def dump_ndim_rmse(name1, name2, model1, model2, vocab1, vocab2, ndim, dirpath,
                   scale):
    z, t, _ = aligner.align_vocab(model1, model2, vocab1, vocab2)
    assert z.shape == t.shape
    results = []
    print('Computing RMSE on {} and {} for every set of {} dims...'
          .format(name1, name2, ndim))
    for idx in tqdm(range(z.shape[1])):
        if idx % ndim == 0:
            if idx + ndim > z.shape[1]:
                break
            m1 = z[:, idx:idx+ndim]
            m2 = t[:, idx:idx+ndim]
            assert m1.shape[1] == m2.shape[1] == ndim
            results.append((idx, get_rmse(m1, m2) * scale))
    output_filepath = os.path.join(
        dirpath, '{}-{}-n{}.dat'.format(name1, name2, ndim))
    print('Saving results to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_str:
        for idx, rmse in results:
            print('{}\t{}'.format(idx, rmse), file=output_str)


def print_batch_results(rmse, xp_results_filepath):
    with open(xp_results_filepath, 'w', encoding='utf-8') as out_str:
        print('Printing RMSE results to file...')
        print('ALIGNMENT RMSE * 10^-4', file=out_str)
        print('\\oanc & {0:.2f} $\\pm$ {0:.2f} &  &  &  & \\\\'.format(
            np.mean(rmse['oanc']['enwiki07']),
            np.std(rmse['oanc']['enwiki07'])), file=out_str)
        print('\\wikitwo & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} &  &  & \\\\'.format(
            np.mean(rmse['enwiki2']['enwiki07']),
            np.std(rmse['enwiki2']['enwiki07']),
            np.mean(rmse['enwiki2']['oanc']),
            np.std(rmse['enwiki2']['oanc'])), file=out_str)
        print('\\acl & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {} &  & \\\\'.format(
            np.mean(rmse['acl']['enwiki07']),
            np.std(rmse['acl']['enwiki07']),
            np.mean(rmse['acl']['oanc']),
            np.std(rmse['acl']['oanc']),
            np.mean(rmse['acl']['enwiki2']),
            np.std(rmse['acl']['enwiki2'])), file=out_str)
        print('\\wikifour & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & \\\\'.format(
            np.mean(rmse['enwiki4']['enwiki07']),
            np.std(rmse['enwiki4']['enwiki07']),
            np.mean(rmse['enwiki4']['oanc']),
            np.std(rmse['enwiki4']['oanc']),
            np.mean(rmse['enwiki4']['enwiki2']),
            np.std(rmse['enwiki4']['enwiki2']),
            np.mean(rmse['enwiki4']['acl']),
            np.std(rmse['enwiki4']['acl'])), file=out_str)
        print('\\bnc & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} & {0:.2f} $\\pm$ {0:.2f} \\\\'.format(
            np.mean(rmse['bnc']['enwiki07']),
            np.std(rmse['bnc']['enwiki07']),
            np.mean(rmse['bnc']['oanc']),
            np.std(rmse['bnc']['oanc']),
            np.mean(rmse['bnc']['enwiki2']),
            np.std(rmse['bnc']['enwiki2']),
            np.mean(rmse['bnc']['acl']),
            np.std(rmse['bnc']['acl']),
            np.mean(rmse['bnc']['enwiki4']),
            np.std(rmse['bnc']['enwiki4'])), file=out_str)


def print_results(rmse, sim, xp_results_filepath):
    with open(xp_results_filepath, 'w', encoding='utf-8') as out_str:
        print('Printing RMSE results to file...')
        print('ALIGNMENT RMSE * 10^-4', file=out_str)
        print('\\oanc & {} &  &  &  & \\\\'.format(
            rmse['oanc']['enwiki07']), file=out_str)
        print('\\wikitwo & {} & {} &  &  & \\\\'.format(
            rmse['enwiki2']['enwiki07'], rmse['enwiki2']['oanc']),
              file=out_str)
        print('\\acl & {} & {} & {} &  & \\\\'.format(
            rmse['acl']['enwiki07'], rmse['acl']['oanc'],
            rmse['acl']['enwiki2']), file=out_str)
        print('\\wikifour & {} & {} & {} & {} & \\\\'.format(
            rmse['enwiki4']['enwiki07'], rmse['enwiki4']['oanc'],
            rmse['enwiki4']['enwiki2'], rmse['enwiki4']['acl']), file=out_str)
        print('\\bnc & {} & {} & {} & {} & {} \\\\'.format(
            rmse['bnc']['enwiki07'], rmse['bnc']['oanc'],
            rmse['bnc']['enwiki2'], rmse['bnc']['acl'],
            rmse['bnc']['enwiki4']), file=out_str)
        print('Printing SIM results to file...')
        print('MODEL\tMEN SPR\tMEN RATIO\tSIMLEX SPR\tSIMLEX RATIO', file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'ENWIKI07', sim['enwiki07']['men']['spr'],
            sim['enwiki07']['men']['ratio'], sim['enwiki07']['simlex']['spr'],
            sim['enwiki07']['simlex']['ratio']), file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'OANC', sim['oanc']['men']['spr'],
            sim['oanc']['men']['ratio'], sim['oanc']['simlex']['spr'],
            sim['oanc']['simlex']['ratio']), file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'ENWIKI2', sim['enwiki2']['men']['spr'],
            sim['enwiki2']['men']['ratio'], sim['enwiki2']['simlex']['spr'],
            sim['enwiki2']['simlex']['ratio']), file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'ACL', sim['acl']['men']['spr'],
            sim['acl']['men']['ratio'], sim['acl']['simlex']['spr'],
            sim['acl']['simlex']['ratio']), file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'ENWIKI4', sim['enwiki4']['men']['spr'],
            sim['enwiki4']['men']['ratio'], sim['enwiki4']['simlex']['spr'],
            sim['enwiki4']['simlex']['ratio']), file=out_str)
        print('{}\t{}\t{}\t{}\t{}'.format(
            'BNC', sim['bnc']['men']['spr'],
            sim['bnc']['men']['ratio'], sim['bnc']['simlex']['spr'],
            sim['bnc']['simlex']['ratio']), file=out_str)


def assert_consistancy_sim_results(sim, name, model, vocab):
    men_spr = evaluator.evaluate_distributional_space(
        model, vocab, 'men', 'spr', 'numpy', 'cosine', 0)[0]
    assert men_spr == sim[name]['men']['spr']
    m_cov_pairs, m_pairs = dutils.get_dataset_coverage('men', vocab)
    men_ratio = (m_cov_pairs / m_pairs) * 100
    assert men_ratio == sim[name]['men']['ratio']
    simlex_spr = evaluator.evaluate_distributional_space(
        model, vocab, 'simlex', 'spr', 'numpy', 'cosine', 0)[0]
    assert simlex_spr == sim[name]['simlex']['spr']
    s_cov_pairs, s_pairs = dutils.get_dataset_coverage('simlex', vocab)
    simlex_ratio = (s_cov_pairs / s_pairs) * 100
    assert simlex_ratio == sim[name]['simlex']['ratio']


def get_rmse(A, B):
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    V = matrixor.apply_absolute_orientation_with_scaling(B, A)
    rmse1 = metrix.root_mean_square_error(A, T)
    rmse2 = metrix.root_mean_square_error(B, V)
    avg = (rmse1 + rmse2) / 2
    return avg


def update_sim_results(sim, name, model, vocab):
    sim[name]['men']['spr'] = evaluator.evaluate_distributional_space(
        model, vocab, 'men', 'spr', 'numpy', 'cosine', 0)[0]
    m_cov_pairs, m_pairs = dutils.get_dataset_coverage('men', vocab)
    sim[name]['men']['ratio'] = (m_cov_pairs / m_pairs) * 100
    sim[name]['simlex']['spr'] = evaluator.evaluate_distributional_space(
        model, vocab, 'simlex', 'spr', 'numpy', 'cosine', 0)[0]
    s_cov_pairs, s_pairs = dutils.get_dataset_coverage('simlex', vocab)
    sim[name]['simlex']['ratio'] = (s_cov_pairs / s_pairs) * 100
    return sim


def get_results(models, scale, rmse, sim, randomize=False):
    for name1, model1, vocab1 in tqdm(models):
        aligned_model1 = model1
        vocab = vocab1
        for name2, model2, vocab2 in models:
            if name1 == name2:
                continue
            assert aligned_model1.shape[1] == model2.shape[1]
            aligned_model1, _, vocab = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
        if not randomize:
            update_sim_results(sim, name1, aligned_model1, vocab)
        for name2, model2, vocab2 in tqdm(models):
            if name1 == name2:
                continue
            A, B, _ = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
            assert A.shape == B.shape
            if not randomize:
                rmse[name1][name2] = get_rmse(A, B) * scale
            else:
                rmse[name1][name2].append(get_rmse(A, B) * scale)
    return rmse, sim


def load_aligned_models(model_names, model_dirpath, start, end, randomize,
                        dims_dirpath, dataset, block_size):
    loaded_models = []
    for name in model_names:
        print('Loading aligned model {}...'.format(name))
        model_path = os.path.join(model_dirpath, '{}-aligned.npy'.format(name))
        vocab_path = os.path.join(model_dirpath, '{}-aligned.vocab'.format(name))
        if dims_dirpath:
            dim_path = os.path.join(dims_dirpath,
                                    '{}-{}.dims'.format(name, dataset))
        else:
            dim_path = None
        model, vocab = dutils.load_model_and_vocab(
            model_path, 'numpy', vocab_path, start=start, end=end,
            shuffle=randomize, dims_filepath=dim_path, block_size=block_size)
        loaded_models.append((name, model, vocab))
    return loaded_models


def load_models(model_names, model_dirpath, start, end):
    loaded_models = []
    for name in model_names:
        print('Loading model {}...'.format(name))
        model_path = os.path.join(model_dirpath, '{}.npy'.format(name))
        vocab_path = os.path.join(model_dirpath, '{}.vocab'.format(name))
        model, vocab = dutils.load_model_and_vocab(
            model_path, 'numpy', vocab_path, start=start, end=end)
        loaded_models.append((name, model, vocab))
    return loaded_models


def launch_xp(model_names, model_dirpath, start, end, scale,
              xp_results_filepath, randomize=False, dims_dirpath=None,
              dataset=None, nruns=0, block_size=0):
    if randomize is True:
        rmse = defaultdict(lambda: defaultdict(list))
        for idx in range(nruns):
            print('Running randomized iter = {}/{}'.format(idx+1, nruns))
            models = load_aligned_models(
                model_names, model_dirpath, start, end, randomize,
                dims_dirpath, dataset, block_size)
            rmse, sim = get_results(models, scale, rmse, None, randomize)
        print_batch_results(rmse, xp_results_filepath)
    else:
        rmse = defaultdict(lambda: defaultdict(dict))
        sim = defaultdict(lambda: defaultdict(dict))
        models = load_aligned_models(
            model_names, model_dirpath, start, end, randomize,
            dims_dirpath, dataset, block_size)
        rmse, sim = get_results(models, scale, rmse, sim, randomize)
        print_results(rmse, sim, xp_results_filepath)
