"""XP1: SVD-TOP-300."""
import os

from tqdm import tqdm

import entropix.utils.data as dutils
import entropix.utils.metrix as metrix
import entropix.core.aligner as aligner
import entropix.core.matrixor as matrixor
import entropix.core.evaluator as evaluator

if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/svd'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/xp1.results'
    START = 0
    END = 300
    print('Running entropix XP#001')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    print('Computing MEN and SIMLEX SPR scores...')
    with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as out_str:
        loaded = []
        for MODEL_NAME in MODEL_NAMES:
            print('Loading model {}...'.format(MODEL_NAME))
            MODEL_PATH = os.path.join(SVD_DIRPATH, '{}.npy'.format(MODEL_NAME))
            VOCAB_PATH = os.path.join(SVD_DIRPATH, '{}.vocab'.format(MODEL_NAME))
            model, vocab = dutils.load_model_and_vocab(
                MODEL_PATH, 'numpy', VOCAB_PATH, start=START, end=END)
            loaded.append((MODEL_NAME.upper(), model, vocab))
        print('ALIGNMENT RMSE', file=out_str)
        for name1, model1, vocab1 in tqdm(loaded):
            aligned_model1 = model1
            vocab = vocab1
            for name2, model2, vocab2 in loaded:
                if name1 == name2:
                    continue
                assert aligned_model1.shape[1] == model2.shape[1]
                aligned_model1, _, vocab = aligner.align_vocab(
                    aligned_model1, model2, vocab, vocab2)
            print('MODEL\tMEN SPR\tMEN RATIO\tSIMLEX SPR\tSIMLEX RATIO', file=out_str)
            m_cov_pairs, m_pairs = dutils.get_dataset_coverage('men', vocab)
            men_spr = evaluator.evaluate_distributional_space(
                aligned_model1, vocab, 'men', 'spr', 'numpy', 'cosine', 0)[0]
            men_ratio = (m_cov_pairs / m_pairs) * 100
            simlex_spr = evaluator.evaluate_distributional_space(
                aligned_model1, vocab, 'simlex', 'spr', 'numpy', 'cosine', 0)[0]
            s_cov_pairs, s_pairs = dutils.get_dataset_coverage('simlex', vocab)
            simlex_ratio = (s_cov_pairs / s_pairs) * 100
            print('{}\t{}\t{}\t{}\t{}'.format(
                name1.upper(), men_spr, men_ratio, simlex_spr, simlex_ratio),
                  file=out_str)
            for name2, model2, vocab2 in loaded:
                if name1 == name2:
                    continue
                A, B, _ = aligner.align_vocab(
                    aligned_model1, model2, vocab, vocab2)
                assert A.shape == B.shape
                print('MODEL\tMEN SPR\tMEN RATIO\tSIMLEX SPR\tSIMLEX RATIO', file=out_str)
                m_cov_pairs, m_pairs = dutils.get_dataset_coverage('men', vocab)
                men_spr = evaluator.evaluate_distributional_space(
                    B, vocab, 'men', 'spr', 'numpy', 'cosine', 0)[0]
                men_ratio = (m_cov_pairs / m_pairs) * 100
                simlex_spr = evaluator.evaluate_distributional_space(
                    B, vocab, 'simlex', 'spr', 'numpy', 'cosine', 0)[0]
                s_cov_pairs, s_pairs = dutils.get_dataset_coverage('simlex', vocab)
                simlex_ratio = (s_cov_pairs / s_pairs) * 100
                print('{}\t{}\t{}\t{}\t{}'.format(
                    name2.upper(), men_spr, men_ratio, simlex_spr, simlex_ratio),
                      file=out_str)
                T = matrixor.apply_absolute_orientation_with_scaling(A, B)
                V = matrixor.apply_absolute_orientation_with_scaling(B, A)
                rmse1 = metrix.root_mean_square_error(A, T)
                rmse2 = metrix.root_mean_square_error(B, V)
                avg = (rmse1 + rmse2) / 2
