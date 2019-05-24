# entropix
Entropy, Zipf's law and distributional semantics.

## Use

### Generate
To generate a raw count matrix from a tokenized corpus, do:
```
entropix generate \
  --corpus abs_corpus_filepath \
  --min-count frequency_threshold \
  --win-size window_size \
  --output abs_output_dirpath
```

If the `--output` parameter is not set, the output files will be saved to the corpus directory.
