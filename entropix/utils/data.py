"""Data utils."""
import os
import logging
import random
import math
import csv
from collections import defaultdict

import joblib
import numpy as np
from scipy import sparse, stats
from tqdm import tqdm
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

__all__ = ('load_model_from_npz', 'load_kfold_splits',
           'load_dataset', 'save_fit_vocab', 'load_model_and_vocab')

DATASETS = {
    'ap': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'resources', 'ap.csv'),
    'battig': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'resources', 'battig.csv'),
    'essli': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'resources', 'essli-2008.csv'),
    'men': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'resources', 'MEN_dataset_natural_form_full'),
    'simlex': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'resources', 'SimLex-999.txt'),
    'simverb': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'SimVerb-3500.txt'),
    'sts2012': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'STS2012.full.txt'),
    'ws353': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'resources', 'WS353.combined.txt')
}


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def save_vocab(vocab, output_vocab_filepath):
    with open(output_vocab_filepath, 'w', encoding='utf-8') as output_stream:
        for word, idx in vocab.items():
            print('{}\t{}'.format(idx, word), file=output_stream)


def save_fit_vocab(dataset_idx, vocab, fit_vocab_filepath):
    logger.info('Saving fit vocab to {}'.format(fit_vocab_filepath))
    reverse_vocab = {value: key for key, value in vocab.items()}
    with open(fit_vocab_filepath, 'w', encoding='utf-8') as fit:
        for idx, item_idx in enumerate(dataset_idx):
            print('{}\t{}'.format(idx, reverse_vocab[item_idx]), file=fit)


def _load_word_pairs_and_sim(dataset):
    """Load word pairs and similarity from a given dataset."""
    if dataset not in ['men', 'simlex', 'simverb', 'sts2012', 'ws353']:
        raise Exception('Unsupported dataset {}'.format(dataset))
    left = []
    right = []
    sim = []
    with open(DATASETS[dataset], 'r', encoding='utf-8') as data_stream:
        for line in data_stream:
            if dataset == 'sts2012':
                line = line.strip().split('\t')
                left.append([x.lower() for x in line[0].split()])
                right.append([x.lower() for x in line[1].split()])
                sim.append(float(line[2]))
            else:
                line = line.rstrip('\n')
                items = line.split()
                left.append(items[0])
                right.append(items[1])
                if dataset in ['men', 'ws353']:
                    sim.append(float(items[2]))
                else:
                    sim.append(float(items[3]))
    return left, right, sim


def load_model_from_npz(model_filepath):
    """Load sparse scipy matrix from .npz filepath"""
    return sparse.load_npz(model_filepath)


def _load_vocab_from_txt_file(vocab_filepath):
    """Load word_to_idx dict mapping from .vocab filepath"""
    word_to_idx = {}
    logger.info('Loading vocabulary from {}'.format(vocab_filepath))
    with open(vocab_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            word_to_idx[linesplit[1]] = int(linesplit[0])
    return word_to_idx


def load_words_set(wordlist_filepath):
    """Load words from a file into a set"""
    words = set()
    with open(wordlist_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            words.add(line.strip().lower())
    return words


def load_dimensions_list(dimensions_filepath):
    ret = []
    with open(dimensions_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linestrip = line.strip()
            ret.append(int(linestrip))
    return ret


def _load_idx_and_sim(left, right, sim, vocab, dataset, shuffle):
    """Load discretized features and similarities for a given dataset.

    Will retain only words that are in the vocabulary.
    Will filter sim accordingly. Will normalize sim between [0, 1]
    """
    left_idx = []
    right_idx = []
    f_sim = []
    if dataset in ['men', 'simlex', 'simverb', 'ws353']:
        for l, r, s in zip(left, right, sim):
            if l in vocab and r in vocab:
                left_idx.append(vocab[l])
                right_idx.append(vocab[r])
                if dataset == 'men':  # men has sim in [0, 50]
                    f_sim.append(s/50)
                else:  # all other datasets have sim in [0, 10]
                    f_sim.append(s/10)
    # TODO: remove hardcoded values?
    if dataset == 'sts2012':
        for l, r, s in zip(left, right, sim):
            l_in_word_to_idx = [x for x in l if x in vocab]
            r_in_word_to_idx = [x for x in r if x in vocab]
            if len(l_in_word_to_idx)/len(l) > 0.85 and\
               len(r_in_word_to_idx)/len(r) > 0.85:
                left_idx.append([vocab[x] for x in l_in_word_to_idx])
                right_idx.append([vocab[x] for x in r_in_word_to_idx])
                f_sim.append(s)
    if shuffle:
        shuffled_zip = list(zip(left_idx, right_idx, f_sim))
        random.shuffle(shuffled_zip)
        unz_shuffle = list(zip(*shuffled_zip))
        return list(unz_shuffle[0]), list(unz_shuffle[1]), list(unz_shuffle[2])
    return left_idx, right_idx, f_sim


def _load_categories_to_words_dict(dataset, vocab):
    if dataset not in ['ap', 'battig', 'essli']:
        raise Exception('Invalid concept categorization dataset: {}'
                        .format(dataset))
    categories_to_words = defaultdict(list)
    with open(DATASETS[dataset], newline='') as csv_stream:
        next(csv_stream)
        reader = csv.reader(csv_stream, quotechar='|')
        for row in reader:
            if row[2] and row[2] in vocab:
                categories_to_words[row[1]].append(row[2])
    return categories_to_words


def load_dataset(dataset, vocab):
    """Load dataset for concept categorization or word similarity."""
    if dataset not in ['ap', 'battig', 'essli', 'men', 'simlex', 'simverb',
                       'ws353']:
        raise Exception('Unsupported dataset: {}'.format(dataset))
    if dataset in ['men', 'simlex', 'simverb', 'ws353']:
        left, right, sim = _load_word_pairs_and_sim(dataset)
        return _load_idx_and_sim(left, right, sim, vocab, dataset,
                                 shuffle=False)
    return _load_categories_to_words_dict(dataset, vocab)


def _load_kfold_splits_dict(left_idx, right_idx, sim, kfold_size, dev_type,
                            dataset, output_logpath):
    if dev_type not in ['nodev', 'regular']:
        raise Exception('Unsupported dev_type = {}'.format(dev_type))
    kfold_dict = defaultdict(defaultdict)
    len_test_set = max(math.floor(len(sim)*kfold_size), 1)
    fold = 1
    max_num_fold = math.floor(len(sim) / len_test_set)
    with open('{}.folds.log'.format(output_logpath), 'w', encoding='utf-8') as folds_log:
        print('{} avg sim = {}'.format(dataset, np.mean(sim)), file=folds_log)
        print('{} std sim = {}'.format(dataset, np.std(sim)), file=folds_log)
        while fold <= max_num_fold:
            test_start_idx = (fold-1)*len_test_set
            test_end_len = test_start_idx + len_test_set
            test_end_idx = test_end_len if test_end_len <= len(sim) else len(sim)
            test_split_idx_set = set(range(test_start_idx, test_end_idx))
            kfold_dict[fold]['test'] = {
                'left_idx': left_idx[test_start_idx:test_end_idx],
                'right_idx': right_idx[test_start_idx:test_end_idx],
                'sim': sim[test_start_idx:test_end_idx]
            }
            train_split_idx_set = set(range(0, len(sim))) - test_split_idx_set
            if dev_type == 'nodev':
                train_split_idx_list = sorted(list(train_split_idx_set))
                kfold_dict[fold]['train'] = {
                    'left_idx': [left_idx[idx] for idx in train_split_idx_list],
                    'right_idx': [right_idx[idx] for idx in train_split_idx_list],
                    'sim': [sim[idx] for idx in train_split_idx_list]
                }
            elif dev_type == 'regular':
                dev_split_idx_set = set(random.sample(train_split_idx_set,
                                                      len(test_split_idx_set)))
                train_split_idx_set = train_split_idx_set - dev_split_idx_set
                dev_split_idx_list = sorted(list(dev_split_idx_set))
                train_split_idx_list = sorted(list(train_split_idx_set))
                kfold_dict[fold]['dev'] = {
                    'left_idx': [left_idx[idx] for idx in dev_split_idx_list],
                    'right_idx': [right_idx[idx] for idx in dev_split_idx_list],
                    'sim': [sim[idx] for idx in dev_split_idx_list]
                }
                kfold_dict[fold]['train'] = {
                    'left_idx': [left_idx[idx] for idx in train_split_idx_list],
                    'right_idx': [right_idx[idx] for idx in train_split_idx_list],
                    'sim': [sim[idx] for idx in train_split_idx_list]
                }
            print('fold {} train avg sim = {}'.format(
                fold, np.mean(kfold_dict[fold]['train']['sim'])),
                  file=folds_log)
            print('fold {} train std sim = {}'.format(
                fold, np.std(kfold_dict[fold]['train']['sim'])),
                  file=folds_log)
            print('fold {} test avg sim = {}'.format(
                fold, np.mean(kfold_dict[fold]['test']['sim'])),
                  file=folds_log)
            print('fold {} test std sim = {}'.format(
                fold, np.std(kfold_dict[fold]['test']['sim'])), file=folds_log)
            fold += 1
    return kfold_dict


def load_kfold_splits(vocab, dataset, kfold_size, dev_type, output_logpath):
    """Return a kfold train/test dict.

    The dict has the form dict[kfold_num] = {train_dict, test_dict} where
    train_dict = {left_idx, right_idx, sim}
    the train/test division follows kfold_size expressed as a precentage
    of the total dataset dedicated to testing.
    kfold_size should be a float > 0 and <= 0.5
    """
    _left, _right, _sim = _load_word_pairs_and_sim(dataset)
    left_idx, right_idx, f_sim = _load_idx_and_sim(
        _left, _right, _sim, vocab, dataset, shuffle=True)
    if kfold_size == 0:
        return {
            1: {
                'train': {
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'sim': f_sim
                },
                'test': {
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'sim': f_sim
                }
            }
        }
    return _load_kfold_splits_dict(left_idx, right_idx, f_sim, kfold_size,
                                   dev_type, dataset, output_logpath)


def load_model_and_vocab(model_filepath, model_type, vocab_filepath=None,
                         singvalues_filepath=None, sing_alpha=None, start=None,
                         end=None, dims_filepath=None, shuffle=False,
                         randomize=False, randtype=None, normloc=None,
                         normscale=None):
    """Load a gensim or scipy/scikit-learn SVD, ICA or NMF model.

    Expects {NUMPY, ICA, NMF, SCIPY} models to have been generated by
    `entropix {svd, ica, nmf, generate}'.
    """
    logger.info('Loading model and vocabulary...')
    if model_type not in ['gensim', 'numpy', 'ica', 'nmf', 'txt', 'scipy']:
        raise Exception('Unsupported model model-type {}'.format(model_type))
    if (shuffle or randomize) and not model_type == 'numpy':
        raise Exception('--shuffle or --randomize only supported with --type numpy')
    if (shuffle or randomize) and dims_filepath:
        raise Exception('Cannot specify both --dims and --shuffle')
    if randomize and randtype not in ['uniform', 'normal']:
        raise Exception('Unsupported --randtype {}. You need  to specify '
                        '--randtype with --randomize'.format(randtype))
    if randtype == 'normal' and (normloc is None or normscale is None):
        raise Exception('You need to specify --normloc and --normscale with '
                        '--randtype normal')
    if model_type == 'gensim':
        wv = Word2Vec.load(model_filepath).wv
        logger.info('model size = {}'.format(wv.vector_size))
        vocab = {word: idx for idx, word in enumerate(wv.wv.vocab.keys())}
        logger.info('vocab size = {}'.format(len(vocab)))
        model = np.empty(shape=(len(vocab), wv.vector_size))
        for word, idx in vocab.items():
            model[idx] = wv[word]
    else:
        if model_type == 'scipy':
            logger.info('Loading raw count-based model from .npz file...')
            vocab = _load_vocab_from_txt_file(vocab_filepath)
            model = load_model_from_npz(model_filepath)
        elif model_type == 'txt':
            vocab = {}
            model = None
            num_rows = 0
            vector_size = 0
            logger.info('Counting number of lines in txt file...')
            with open(model_filepath, 'r', encoding='latin1') as model_txt:
                for idx, line in enumerate(model_txt):
                    num_rows += 1
                    if num_rows == 1:
                        tokens = line.strip().split(' ', 1)
                        vector_size = len(tokens[1].split(' '))
            model = np.empty((num_rows, vector_size), dtype=np.float32)
            logger.info('Initializing matrix of shape = {}'
                        .format(model.shape))
            logger.info('Filling in matrix entries...')
            with open(model_filepath, 'r', encoding='latin1') as model_txt:
                for idx, line in tqdm(enumerate(model_txt), total=num_rows):
                    tokens = line.strip().split(' ', 1)
                    vocab[tokens[0]] = idx
                    model[idx] = np.fromstring(tokens[1], sep=' ',
                                               dtype=np.float32)
        else:
            vocab = _load_vocab_from_txt_file(vocab_filepath)
            if model_type == 'numpy':
                if randomize:
                    size = end-start
                    logger.warning('Generating matrix of random vectors with '
                                   'shape = [{},{}] using a {} distribution'
                                   .format(len(vocab), size, randtype))
                    if randtype == 'uniform':
                        model = np.random.uniform(-1, 1, [len(vocab), size])
                    elif randtype == 'normal':
                        distrib = get_truncated_normal(
                            mean=normloc, sd=normscale, low=-1, upp=1)
                        model = distrib.rvs(len(vocab), size)
                    return model, vocab
                logger.info('Loading numpy model...')
                singvectors = np.load(model_filepath)
                if dims_filepath:
                    dims = []
                    with open(dims_filepath, 'r', encoding='utf-8') as dims_stream:
                        for line in dims_stream:
                            dims.append(int(line.strip()))
                    logger.info('Sampling model with {} dimensions = {}'
                                .format(len(dims), dims))
                    logger.info('Reducing model to {} dims'.format(len(dims)))
                if sing_alpha is not None and sing_alpha != 0:
                    logger.info(
                        'Loading model with singalpha = {} from singvalues {}'
                        .format(sing_alpha, singvalues_filepath))
                    singvalues = np.load(singvalues_filepath)
                if start is not None and end is not None and not shuffle:
                    logger.info('Reducing model to start = {} and end = {}'
                                .format(start, end))
                    singvectors = singvectors[:, start:end]
                    if sing_alpha is not None and sing_alpha != 0:
                        singvalues = singvalues[start:end]
                if dims_filepath and dims:
                    singvectors = singvectors[:, dims]
                    if sing_alpha is not None and sing_alpha != 0:
                        singvalues = singvalues[dims]
                if sing_alpha is None or sing_alpha == 0:
                    model = singvectors
                elif sing_alpha == 1:
                    model = np.matmul(singvectors, np.diag(singvalues))
                else:
                    model = np.matmul(singvectors, np.diag(np.power(singvalues,
                                                                    sing_alpha)))
                if shuffle:
                    logger.warning('Shuffle mode ON: will randomly select dimensions')
                    randims = list(range(model.shape[1]))
                    random.shuffle(randims)
                    if start is not None and end is not None:
                        randims = randims[start:end]
                    model = model[:, randims]
            else:
                if model_type == 'ica':
                    logger.info('Loading scikit-learn ICA model...')
                else:
                    logger.info('Loading scikit-learn NMF model...')
                model = joblib.load(model_filepath)
        logger.info('model size = {}'.format(model.shape))
    return model, vocab
