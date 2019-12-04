"""Compare RMSE every N set of dims on the SVD of two distinct corpora."""

from tqdm import tqdm

import entropix.utils.data as dutils
import entropix.utils.metrix as metrix
import entropix.core.aligner as aligner
import entropix.core.matrixor as matrixor

if __name__ == '__main__':
    SVD1 = '/Users/akb/Github/entropix/models/svd/bnc.mincount-30.win-2.npy'
    SVD2 = '/Users/akb/Github/entropix/models/svd/enwiki4.mincount-30.win-2.npy'
    VOCAB1 = '/Users/akb/Github/entropix/models/svd/bnc.mincount-30.win-2.vocab'
    VOCAB2 = '/Users/akb/Github/entropix/models/svd/enwiki4.mincount-30.win-2.vocab'
    NDIM = 30
    OUTPUT_FILEPATH = '/Users/akb/Gitlab/frontiers/data/bnc-enwiki4-n30.dat'
    print('Loading model from {}'.format(SVD1))
    model1, vocab1 = dutils.load_model_and_vocab(
        SVD1, 'numpy', VOCAB1)
    print('Loading model from {}'.format(SVD2))
    model2, vocab2 = dutils.load_model_and_vocab(
        SVD2, 'numpy', VOCAB2)
    assert model1.shape[1] == model2.shape[1]
    print('Aligning models vocabulary...')
    aligned_model1, aligned_model2, vocab = aligner.align_vocab(
        model1, model2, vocab1, vocab2)
    assert aligned_model1.shape[0] == aligned_model2.shape[0]
    print('vocab size = {}'.format(aligned_model1.shape[0]))
    results = []
    print('Computing RMSE for every set of {} dims...'.format(NDIM))
    for idx in tqdm(range(model1.shape[1])):
        if idx % NDIM == 0:
            if idx + NDIM > model1.shape[1]:
                break
            m1 = aligned_model1[:, idx:idx+NDIM]
            m2 = aligned_model2[:, idx:idx+NDIM]
            assert m1.shape[1] == m2.shape[1] == NDIM
            T = matrixor.apply_absolute_orientation_with_scaling(m1, m2)
            X = matrixor.apply_absolute_orientation_with_scaling(m2, m1)
            rmse1 = metrix.root_mean_square_error(m1, T)
            rmse2 = metrix.root_mean_square_error(m2, X)
            rmse = (rmse1 + rmse2) / 2
            results.append((idx, rmse))
    print('Saving results to {}'.format(OUTPUT_FILEPATH))
    with open(OUTPUT_FILEPATH, 'w', encoding='utf-8') as output_str:
        for idx, rmse in results:
            print('{}\t{}'.format(idx, rmse), file=output_str)
