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
To export


## Sample


## Intersect
To intersect the vocabularies of two models generated from two different corpora, run:
```

```

## Align
To align two models via centering + rotation, run:
```

```

## Compare
To compare two aligned models by computing the RMSE, run:
```

```
