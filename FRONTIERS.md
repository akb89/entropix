# Frontiers paper experiments

## Generate
To generate a raw count matrix from a tokenized corpus, run:
```
entropix generate \
--corpus /abs/path/to/corpus/txt/file \
--min-count frequency_threshold \
--win-size window_size \
--output abs/path/to/output/model/directory
```

If the `--output` parameter is not set, the output files will be saved to the corpus directory.

## Weigh
To weigh a raw count model with PPMI, run:
```
entropix weigh \
--model /abs/path/to/raw/count/npz/model \
--output /abs/path/to/output/model/directory \
--weighing-func ppmi
```

## SVD
To apply SVD on a PPMI-weighed model, with k=10000, run:
```
entropix svd \
--model /abs/path/to/ppmi/npz/model \
--dim 10000 \
--which LM  # largest singular values
```

## Export
To export the top 300 dimensions from the SVD singular vectors, do:
```
entropix export \
--model /abs/path/to/svd/lm/model \
--type numpy \
--vocab /abs/path/to/vocab \
--start 0 \
--end 300 \
--output /abs/path/to/output/model
```


## Sample


## Align
To intersect the vocabularies of two models generated from two different corpora, run:
```
entropix align \
--model1 /abs/path/to/np/model1 \
--model2 /abs/path/to/np/mode2 \
--vocab1 /abs/path/to/vocab1 \
--vocab2 /abs/path/to/vocab2 \
--outputname model1-model2-align
```

## Transform
To align two models via centering + rotation and get the RMSE, run:
```

```
