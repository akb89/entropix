"""Generate plots and analysis from results"""

import os
import glob
import entropix
import entropix.core.extractor as extractor
import entropix.core.visualizer as visualizer

if __name__ == '__main__':
    print('Running entropix XPVIZ#001')

    OUTPUT_DIRPATH = '/tmp/'
    assert os.path.exists(OUTPUT_DIRPATH)

    MODELS_DIRPATH = '/home/ludovica/Scaricati/ppmi/'
    assert os.path.exists(MODELS_DIRPATH)

    vocab_filepath = '/home/ludovica/Scaricati/enwiki.20190120.mincount-300.win-2.vocab'

    # Issue 1:
    # Relevant words for each dimension. We're taking the 10 highest positive
    # values and the 10 lowest negative values

    print('Issue1')

    for sing_vectors_filepath in os.listdir(MODELS_DIRPATH):
        sing_vectors_filepath = os.path.join(MODELS_DIRPATH, sing_vectors_filepath)
        extractor.extract_top_participants(
            OUTPUT_DIRPATH, sing_vectors_filepath, vocab_filepath, 20, True)


    # Issue 2:
    # Which dimensions are semantically more coherent?

    # print('Issue2')
    # visualizer.visualize_cosine_similarity(
    #     OUTPUT_DIRPATH,
    #     '{}enwiki.20190120.mincount-300.win-2.ppmi.k1000.top-20.pos'.format(OUTPUT_DIRPATH),
    #     '{}enwiki.20190120.mincount-300.win-2.ppmi.k1000.top-20.neg'.format(OUTPUT_DIRPATH),
    #     '{}enwiki.20190120.mincount-300.win-2.ppmi.k1000.singvectors.npy'.format(MODELS_DIRPATH),
    #     vocab_filepath)
    #
    # visualizer.visualize_cosine_similarity(
    #     OUTPUT_DIRPATH,
    #     '{}enwiki.20190120.mincount-300.win-5.ppmi.k1000.top-20.pos'.format(OUTPUT_DIRPATH),
    #     '{}enwiki.20190120.mincount-300.win-5.ppmi.k1000.top-20.neg'.format(OUTPUT_DIRPATH),
    #     '{}enwiki.20190120.mincount-300.win-5.ppmi.k1000.singvectors.npy'.format(MODELS_DIRPATH),
    #     vocab_filepath)

    # Issue 3:
    # When a limited number of dimensions is set, what dimensions are selected?

#    print('Issue3')

#    pref = '/home/ludovica/Desktop/entropix_analysis/sampled/win2/'
#    best_models_filepaths = [
#        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-10.shuffle.txt',
#        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt',
#        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt']
#    n_best_models = 10000

#    limited_models_filepaths = [
#        pref+'k5000/enwiki.20190120.mincount-300.win-2.ppmi.k5000.men.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
#        pref+'k5000/enwiki.20190120.mincount-300.win-2.ppmi.k5000.simlex.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
#        pref+'k5000/enwiki.20190120.mincount-300.win-2.ppmi.k5000.simverb.sampledims.mode-limit.d-30.start-0.end-0.final.txt']
#    n_limited_models = 5000

#    visualizer.visualize_boxplot_per_dataset(
#        OUTPUT_DIRPATH,
#        n_best_models,
#        best_models_filepaths)
    #
#    visualizer.visualize_boxplot_per_dataset(
#        OUTPUT_DIRPATH,
#        n_limited_models,
#        limited_models_filepaths)

    pref = '/home/ludovica/Desktop/entropix_analysis/sampled/win2/'
    men_shuffled = [
        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.txt',
        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-10.shuffle.txt'
    ]
#    visualizer.visualize_boxplot_per_shuffle(
#        OUTPUT_DIRPATH,
#        10000,
#        men_shuffled)

    pref = '/home/ludovica/Desktop/entropix_analysis/sampled/win2/'
    simlex_shuffled = [
        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.iter-1.reduce.step-6.txt',
        pref+'k10000/enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt'
    ]
#    visualizer.visualize_boxplot_per_shuffle(
#        OUTPUT_DIRPATH,
#        10000,
#        simlex_shuffled)




    # Issue 4:
    # Is there a correlation between the selected dimension and its internal semantic coherence?
    # i.e. are more semantically coherent dimensions more often selected or not?

    # Issue 5:
    # Does the shuffling influence the distribution of dimension?

    # Issue 6:
