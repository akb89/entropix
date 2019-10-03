# entropix
Entropy, Zipf's law and distributional semantics.

## Use

### Generate
To generate a raw count matrix from a tokenized corpus, run:
```
entropix generate \
  --corpus /abs/path/to/corpus/txt/file \
  --min-count frequency_threshold \
  --win-size window_size \
  --output abs/path/to/output/model/directory
```

If the `--output` parameter is not set, the output files will be saved to the corpus directory.

### Weigh
To weigh a raw count model with PPMI, run:
```
entropix weigh \
  --model /abs/path/to/raw/count/npz/model \
  --output /abs/path/to/output/model/directory \
  --weighing-func ppmi
```

### SVD
To apply SVD on a PPMI-weighed model, with k=10000, run:
```
entropix svd \
  --model /abs/path/to/ppmi/npz/model \
  --dim 10000 \
  --which LM  # largest singular values
```

## Warning
For informativeness, np.dot in `prob_values = np.exp(np.dot(l1, self._model.trainables.syn1neg.T))` using multithreading by default with openblas. Need to run the code with `env OMP_NUM_THREADS=1 entropix generate...`
