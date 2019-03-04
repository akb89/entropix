"""Generate plots and analysis from results"""

import os
import glob
import entropix
from datetime import datetime
import entropix.core.extractor as extractor
import entropix.core.visualizer as visualizer

if __name__ == '__main__':
    print('Running entropix XPVIZ#002')

    OUTPUT_DIRPATH = '/home/ludovica/Desktop/entropix_analysis/'
    assert os.path.exists(OUTPUT_DIRPATH)

    MODELS_DIRPATH = '/home/ludovica/entropix_models/'
    assert os.path.exists(MODELS_DIRPATH)

    vocab_filepath = '/home/ludovica/Downloads/enwiki.20190120.mincount-300.win-2.vocab'

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = [
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.sts2012.sampledims.mode-seq.niter-2.start-0.end-0.keep.iter-2.reduce.step-3.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simlex.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simverb.sampledims.mode-limit.d-30.start-0.end-0.final.txt']
    n_best_models = 1000
#    visualizer.visualize_boxplot_per_dataset(
#            OUTPUT_DIRPATH,
#            n_best_models,
#            best_models_filepaths,
#            fname='swarmplot.k1000.d30.sts2012.men.simlex.simverb.png')

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = [
#        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.sts2012.sampledims.mode-seq.niter-2.start-0.end-0.keep.iter-2.reduce.step-3.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k5000.men.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k5000.simlex.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k5000.simverb.sampledims.mode-limit.d-30.start-0.end-0.final.txt']
    n_best_models = 5000
#    visualizer.visualize_boxplot_per_dataset(
#            OUTPUT_DIRPATH,
#            n_best_models,
#            best_models_filepaths,
#            fname='swarmplot.k5000.d30.men.simlex.simverb.png')

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = [
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.sts2012.sampledims.mode-seq.niter-2.start-0.end-0.keep.iter-2.reduce.step-3.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simlex.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simverb.sampledims.mode-limit.d-30.start-0.end-0.final.txt']
    wordlist = '/home/ludovica/Desktop/entropix_analysis/enwiki.20190120.mincount-300.win-2.ppmi.k1000.top-20.abs'
#    visualizer.visualize_clustermap_lexoverlap(
#            OUTPUT_DIRPATH,
#            best_models_filepaths,
#            wordlist,
#            fname='heatmap.k1000.d30.png',
#            xticks=True)

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = [
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.sts2012.sampledims.mode-seq.niter-3.start-0.end-0.keep.iter-3.reduce.step-3.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.mode-seq.rate-10.keep.iter-1.reduce.step-6.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simlex.sampledims.mode-seq.niter-2.start-0.end-0.keep.iter-2.reduce.step-3.txt',
        pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-4.txt']
    wordlist = '/home/ludovica/Desktop/entropix_analysis/enwiki.20190120.mincount-300.win-2.ppmi.k1000.top-20.abs'
#    visualizer.visualize_clustermap_lexoverlap(
#            OUTPUT_DIRPATH,
#            best_models_filepaths,
#            wordlist,
#            fname='heatmap.k1000.best.png')

#    visualizer.visualize_boxplot_per_dataset(
#            OUTPUT_DIRPATH,
#            1000,
#            best_models_filepaths,
#            fname='swarmplot.k1000.best.sts2012.men.simlex.simverb.png')



    pref = '/home/ludovica/entropix_sampleddims/'

#    best_models_filepaths = {
#        'linear': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.keep.iter-1.reduce.step-6.txt',
#        'shuffled': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.keep.iter-1.reduce.step-7.shuffle.txt',
#    }
#    nmax = 1000
#    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, nmax, 'men.k1000.selection.png')

    best_models_filepaths = {
        'linear': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.txt',
        'shuffled': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.shuffle.txt',
    }
    nmax = 10000
    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, nmax, 'men.k10000.selection.png')

    best_models_filepaths = {
        'linear': pref+'sts/enwiki.20190120.mincount-300.win-2.ppmi.k10000.sts2012.sampledims.mode-seq.niter-5.start-0.end-0.keep.iter-5.reduce.step-3.txt',
        'shuffled': pref+'sts/enwiki.20190120.mincount-300.win-2.ppmi.k10000.sts2012.sampledims.mode-seq.niter-5.start-0.end-0.keep.shuffled.iter-5.reduce.step-1.txt',
    }
    nmax = 10000
    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, nmax, 'sts2012.k10000.selection.png')

    best_models_filepaths = {
        'men': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.shuffle.txt',
        'simlex': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-9.txt',
        'simverb': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt',
    }
    nmax = 10000
    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, nmax, 'barcode.bias.k10000.selection.png')
    visualizer.visualize_boxplot_per_dataset(OUTPUT_DIRPATH, nmax, best_models_filepaths, 'boxplot.limit30.k10000.selection.png')

    best_models_filepaths = {
        'men': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.men.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        'simlex': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simlex.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
        'simverb': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k1000.simverb.sampledims.mode-limit.d-30.start-0.end-0.final.txt',
    }
    nmax = 1000
    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, nmax, 'barcode.limit30.k1000.selection.png')
    visualizer.visualize_boxplot_per_dataset(OUTPUT_DIRPATH, nmax, best_models_filepaths, 'boxplots.limit30.k1000.selection.png')



    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = {
        'men': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.txt',
        'simlex': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-9.txt',
        'simverb': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt'}

    visualizer.visualize_boxplot_per_dataset(OUTPUT_DIRPATH, 10000, best_models_filepaths, 'boxplot.men.simlex.sts.10000.best.png')

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = {
        'men': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.txt',
        'simlex': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-9.txt',
        'simverb': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt'}

    visualizer.visualize_barcode(OUTPUT_DIRPATH, best_models_filepaths, 10000, 'barcode.men.simlex.sts.10000.best.png')

    pref = '/home/ludovica/entropix_sampleddims/'
    best_models_filepaths = {
        'men': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.men.sampledims.keep.iter-1.reduce.step-9.txt',
        'simlex': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simlex.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-9.txt',
        'simverb': pref+'enwiki.20190120.mincount-300.win-2.ppmi.k10000.simverb.sampledims.mode-seq.niter-1.start-0.end-0.keep.shuffled.iter-1.reduce.step-10.txt'}

#    visualizer.visualize_coverage(OUTPUT_DIRPATH, best_models_filepaths, 10000, 'coverage.men.simlex.sts.10000.best.png')

    # visualizer.visualize_bottomup_barchart(OUTPUT_DIRPATH, '/home/ludovica/Desktop/entropix_analysis/top-down.csv', 'top-down.png')
    #
    # selected_words = ['cat', 'bunny', 'dog', 'hawk',
    #                   'car', 'vehicle', 'bicycle', 'truck',
    #                   'flowers', 'garden', 'blossom',
    #                   'winter', 'autumn', 'spring', 'summer',
    #                   'red', 'violet', 'color',
    #                   'building', 'construction', 'railway', 'station', 'road',
    #                   'the', 'a', 'one', 'that',
    #                   'some', 'many', 'few']
    #
    # visualizer.visualize_word_space(
    #         OUTPUT_DIRPATH,
    #         selected_words,
    #         model_filepath,
    #         vocab_filepath,
    #         best_models_filepaths,
    #         fname='spaceheatmap.k1000.d30.sts2012.men.simlex.simverb.png')
