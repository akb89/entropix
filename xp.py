import os
import datetime

from tqdm import tqdm

import entropix.utils.data as dutils
import entropix.core.evaluator as evaluator

if __name__ == '__main__':
    MODELS_DIRPATH = '/Users/akb/Github/entropix/models/todo/'
    output_filename = 'results-todo-{}.txt'.format(datetime.datetime.now().timestamp())
    models_filepaths = [os.path.join(MODELS_DIRPATH, filename) for filename in
                        os.listdir(MODELS_DIRPATH) if filename.endswith('.npy')]
    with open(output_filename, 'w', encoding='utf-8') as output_stream:
        print('NAME\tMEN-SPR\tMEN-RMSE\tSIMLEX-SPR\tSIMLEX-RMSE\tSIMVERB-SPR\tSIMVERB-RMSE\tAP\tBATTIG\tESSLI\tDIM', file=output_stream)
        for model_filepath in tqdm(sorted(models_filepaths)):
            vocab_filepath = '{}.vocab'.format(model_filepath.split('.npy')[0])
            model, vocab = dutils.load_model_and_vocab(
                model_filepath, 'numpy', vocab_filepath)
            name = os.path.basename(model_filepath).split('.npy')[0]
            men_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='men', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)
            men_rmse = evaluator.evaluate_distributional_space(
                model, vocab, dataset='men', metric='rmse', model_type='numpy',
                distance='cosine', kfold_size=0)
            simlex_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='simlex', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)
            simlex_rmse = evaluator.evaluate_distributional_space(
                model, vocab, dataset='simlex', metric='rmse', model_type='numpy',
                distance='cosine', kfold_size=0)
            simverb_spr = evaluator.evaluate_distributional_space(
                model, vocab, dataset='simverb', metric='spr', model_type='numpy',
                distance='cosine', kfold_size=0)
            simverb_rmse = evaluator.evaluate_distributional_space(
                model, vocab, dataset='men', metric='rmse', model_type='numpy',
                distance='cosine', kfold_size=0)
            ap = evaluator.evaluate_distributional_space(
                model, vocab, dataset='ap', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            battig = evaluator.evaluate_distributional_space(
                model, vocab, dataset='battig', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            essli = evaluator.evaluate_distributional_space(
                model, vocab, dataset='essli', metric=None, model_type='numpy',
                distance=None, kfold_size=None)
            dim = model.shape[1]
            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                name, men_spr, men_rmse, simlex_spr, simlex_rmse, simverb_spr,
                simverb_rmse, ap, battig, essli, dim), file=output_stream)
