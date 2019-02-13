# entropix
Entropy, Zipf's law and distributional semantics.

## Use

### Generate
To generate the count matrix from a tokenized corpus, do:
```
entropix count \
  --corpus abs_corpus_filepath \
  --min-count frequency_threshold \
  --win-size window_size \
  --output abs_output_dirpath \
```

If the `--output` parameter is not set, the output files will be saved to the corpus directory.

For example, running:
```
entropix count \
  -c /home/USER/corpus.txt \
  -m 50 \
  -w 2 \
  -o /home/USER/output_counts/ \
```
will produce:
* `/home/USER/output_counts/corpus.mincount-50.win-2.npz` -> serialized sparse csr matrix
* `/home/USER/output_counts/corpus.mincount-50.win-2.vocab` -> word-to-index mappings to interpret the matrix dimensions
