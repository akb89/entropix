# Experimental results

## Generate a count-based model from a Wikipedia dump
Window 2
```
/home/kabbach/venv/bin/entropix generate \
  --corpus /home/kabbach/witokit/data/wiki/enwiki.20190120.txt \
  --output /home/kabbach/entropix/models/mincount-30/raw/ \
  --min-count 30 \
  --win-size 2
```

Window 5
```
/home/kabbach/venv/bin/entropix generate \
  --corpus /home/kabbach/witokit/data/wiki/enwiki.20190120.txt \
  --output /home/kabbach/entropix/models/mincount-30/raw/ \
  --min-count 30 \
  --win-size 5
```

## Compute PPMI
Window 2
```
/home/kabbach/venv/bin/entropix weigh \
  --model /home/kabbach/entropix/models/mincount-30/raw/enwiki.20190120.mincount-30.win-2.npz \
  --output /home/kabbach/entropix/models/mincount-30/ppmi/ \
  -w ppmi
```

Window 5
```
/home/kabbach/venv/bin/entropix weigh \
  --model /home/kabbach/entropix/models/mincount-30/raw/enwiki.20190120.mincount-30.win-5.npz \
  --output /home/kabbach/entropix/models/mincount-30/ppmi/ \
  -w ppmi
```

## Reduce via SVD
Window 2
```
/home/kabbach/venv/bin/entropix svd \
  --model /home/kabbach/entropix/models/mincount-30/ppmi/enwiki.20190120.mincount-30.win-2.ppmi.npz \
  --dim 1000
```

Window 5
```
/home/kabbach/venv/bin/entropix svd \
  --model /home/kabbach/entropix/models/mincount-30/ppmi/enwiki.20190120.mincount-30.win-5.ppmi.npz \
  --dim 1000
```

## Baseline
