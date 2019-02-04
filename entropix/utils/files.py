"""Files utils."""

import os
import logging

logger = logging.getLogger(__name__)

__all__ = ('get_input_filepaths')


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
    return os.path.join(dirpath,
                        '{}.cos.gz'.format(os.path.basename(model_filepath)))


def get_cosines_distribution_filepath(dirpath):
    """
    Return the filepath, into the specified folder, to pwcosines.dist.txt file.
    """
    return os.path.join(os.path.join(dirpath, 'pwcosines.dist.txt'))


def get_counts_filepath(corpus_filepath, output_dirpath):
    """
    Return the .counts filepath associated to the corpus filepath passed as
    a parameter, in the output directory passed as a parameter.
    """
    if output_dirpath:
        if corpus_filepath.endswith('.txt'):
            output_filepath = os.path.join(
                output_dirpath, '{}.counts'.format(
                    os.path.basename(corpus_filepath).split('.txt')[0]))
        else:
            output_filepath = os.path.join(
                output_dirpath,
                '{}.counts'.format(os.path.basename(corpus_filepath)))

    return output_filepath


def get_sparsematrix_filepath(output_dirpath, corpus_filepath,
                              min_count, win_size):
    """
    Return the .mincount-[].win-[] filepath associated to the corpus filepath
    passed as a parameter, in the folder passed as a parameter.
    """
    if corpus_filepath.endswith('.txt'):
        output_filepath_matrix = os.path.join(
            output_dirpath, '{}.mincount-{}.win-{}'.format(
                os.path.basename(corpus_filepath).split('.txt')[0], min_count,
                win_size))
    else:
        output_filepath_matrix = os.path.join(
            output_dirpath,
            '{}.mincount-{}.win-{}'.format(os.path.basename(corpus_filepath),
                                           min_count, win_size))
    return output_filepath_matrix
