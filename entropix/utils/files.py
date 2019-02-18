"""Files utils."""

import os
import logging

logger = logging.getLogger(__name__)

__all__ = ('get_input_filepaths', 'get_vocab_filepath', 'get_cosines_filepath',
           'get_cosines_distribution_filepath', 'get_counts_filepath',
           'get_sparsematrix_filepath', 'get_singvalues_filepath',
           'get_singvectors_filepath', 'get_singvalues_filepaths',
           'get_models_filepaths', 'get_topelements_filepath')


def get_models_filepaths(model_dirpath):
    """Return all *.npz files under model directory."""
    return [os.path.join(model_dirpath, filename) for filename in
            os.listdir(model_dirpath) if filename.endswith('npz')]


def get_singvalues_filepaths(model_dirpath):
    """Return all *.singvalues.npy files under model directory."""
    return [os.path.join(model_dirpath, filename) for filename in
            os.listdir(model_dirpath) if filename.endswith('.singvalues.npy')]


def _get_model_basename(model_filepath):
    if model_filepath.endswith('.npy'):
        return model_filepath.split('.npy')[0]
    return model_filepath.split('.npz')[0]


def get_singvectors_filepath(sparse_model_filepath, dim, compact):
    """Return absolute path to singular vectors .npz file."""
    model_basename = _get_model_basename(sparse_model_filepath)
    if compact:
        return '{}.singvectors.npy'.format(model_basename)
    return '{}.k{}.singvectors.npy'.format(model_basename, dim)


def get_singvalues_filepath(sparse_model_filepath, dim, compact):
    """Return absolute path to singular values .npz file."""
    model_basename = _get_model_basename(sparse_model_filepath)
    if compact:
        '{}.singvalues.npy'.format(model_basename)
    return '{}.k{}.singvalues.npy'.format(model_basename, dim)


def get_input_filepaths(dirpath):
    """Return all the files under the specified directory."""
    return [os.path.join(dirpath, filename) for filename in
            os.listdir(dirpath) if not filename.startswith('.')]


def get_vocab_filepath(model_filepath):
    """
    Return the .vocab filepath associated to the model filepath
    passed as a parameter.
    """
    basename_model_filepath = os.path.basename(model_filepath)
    dirname = os.path.dirname(model_filepath)
    vocab_filepath = '{}.vocab'.format(basename_model_filepath)
    if basename_model_filepath.endswith('.npz'):
        vocab_filepath = '{}.vocab'.format(model_filepath[:-len('.npz')])
    vocab_filepath = os.path.join(dirname, vocab_filepath)
    return vocab_filepath


def get_cosines_filepath(dirpath, model_filepath):
    """
    Return the .cos.gz filepath associated to the model filepath
    passed as a parameter, in the folder passed as a paremeter.
    """
    model_filepath_basename = _get_model_basename(os.path.basename(model_filepath))
    return os.path.join(dirpath,
                        '{}.cos.gz'.format(model_filepath_basename))


def get_cosines_distribution_filepath(dirpath, model_filepath):
    """
    Return the filepath, into the specified folder, to pwcosines.dist.txt file.
    """
    model_filepath_basename = _get_model_basename(os.path.basename(model_filepath))
    return os.path.join(os.path.join(dirpath, '{}.pwcosines.dist.txt'.format(model_filepath_basename)))


def get_singvectors_distribution_filepath(dirpath, model_filepath):
    """
    Return the filepath, into the specified folder, to singvectors.dist.txt file.
    """
    model_basename = _get_model_basename(model_filepath)
    return os.path.join(os.path.join(dirpath, '{}.singvectors.dist.txt'
                                     .format(model_basename)))


def get_counts_filepath(corpus_filepath, output_dirpath):
    """
    Return the .counts filepath associated to the corpus filepath passed as
    a parameter, in the output directory passed as a parameter.
    """
    if output_dirpath:
        dirname = output_dirpath
    else:
        dirname = os.path.dirname(corpus_filepath)

    if corpus_filepath.endswith('.txt'):
        output_filepath = os.path.join(
            dirname, '{}.counts'.format(
                os.path.basename(corpus_filepath).split('.txt')[0]))
    else:
        output_filepath = os.path.join(
            dirname,
            '{}.counts'.format(os.path.basename(corpus_filepath)))

    return output_filepath


def get_weightedmatrix_filepath(output_dirpath, model_filepath):
    model_basename = os.path.basename(model_filepath)
    return os.path.join(output_dirpath, '{}.ppmi'.format(model_basename.split('.npz')[0]))


def get_sparsematrix_filepath(output_dirpath, corpus_filepath,
                              min_count, win_size):
    """
    Return the .mincount-[].win-[] filepath associated to the corpus filepath
    passed as a parameter, in the folder passed as a parameter.
    """
    if corpus_filepath.endswith('.txt'):
        output_filepath_matrix = os.path.join(
            output_dirpath, '{}.mincount-{}.win-{}.npz'.format(
                os.path.basename(corpus_filepath).split('.txt')[0], min_count,
                win_size))
    else:
        output_filepath_matrix = os.path.join(
            output_dirpath,
            '{}.mincount-{}.win-{}.npz'.format(
                os.path.basename(corpus_filepath), min_count, win_size))
    return output_filepath_matrix

def _create_tmp_folder(output_dirpath):
    tmp_dirpath = os.path.join(output_dirpath, 'tmp')
    if not os.path.exists(tmp_dirpath):
        logger.info('Creating directory {}'.format(tmp_dirpath))
        os.makedirs(tmp_dirpath)
    return tmp_dirpath

def get_tmp_cosinedist_filepath(output_dirpath, idx):
    tmp_path = _create_tmp_folder(output_dirpath)
    return os.path.join(tmp_path, '{}.cosinesim'.format(idx))

def get_topelements_filepath(output_dirpath, sing_vectors_filepath, N):
    matrix_basename = os.path.basename(sing_vectors_filepath)
    basename = matrix_basename
    if matrix_basename.endswith('.singvectors.npy'):
        basename = matrix_basename.split('.singvectors.npy')[0]

    return os.path.join(output_dirpath, '{}.top-{}.words'.format(basename, N))
