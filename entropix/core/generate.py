"""A basic count-based model using sparse matrices and no ppmi."""
import os
import collections
from datetime import datetime
from scipy import sparse

def count_cooc (fname, window, freqdist, fr_threshold, word_to_idx_map):
    print(fr_threshold)

	final_rows = []
	final_columns = []
	final_data = []
    with open(fname, 'r', encoding='utf-8') as fin:
        for line_num, line in enumerate(fin):
			linesplit = line.strip().split()
			for idx, tok in enumerate(linesplit):
					tok_mapped_index = word_to_idx_map[tok]
					start = 0 if window == 0 else max(0, idx-window)
					end = idx
					while start<end:
						cooctok = linesplit[start]
						cooctok_mapped_index = word_to_idx_map[cooctok]
						final_rows.append(tok_mapped_index)
						final_columns.append(cooctok_mapped_index)
						final_rows.append(cooctok_mapped_index)
						final_columns.append(tok_mapped_index)
						start+=1
			if not line_num%100000:
				print(line_num)


	idx_to_word_map = {k:w for w,k in word_to_idx_map.items() if freqdist[w]>fr_threshold}


	print("reverse freqdist len")
	print(len(idx_to_word_map))

	print("freqdist len")
	print(len(freqdist))


	new_map = {k:inc for k, inc in zip(idx_to_word_map.keys(), range(len(idx_to_word_map)))}


	keep_indexes = []
	idx = 0
	for row, column in zip(final_rows, final_columns):
		if row in idx_to_word_map and column in idx_to_word_map:
			keep_indexes.append(idx)


		idx+=1

	return [1]*len(keep_indexes), [new_map[final_rows[i]] for i in keep_indexes], [new_map[final_columns[i]] for i in keep_indexes], new_map


def compute_word_freqs (fname):

	voc = collections.defaultdict(int)
	word_ind = {}
	latest_ind = 0

	with open(fname) as fin:

		for line in fin:
			linesplit = line.strip().split()

			for tok in linesplit:
				voc[tok]+=1
				if not tok in word_ind:
					word_ind[tok]=latest_ind
					latest_ind+=1

	return voc, word_ind

if __name__ == "__main__":

	_NTHR = 1
	_WINSIZE = 2
	_MINCOUNT = 50

	#~ fname = "enwiki.20190120.txt.sample2.0.head"
	fname = "enwiki.20190120.txt.sample2.0"

	global_word_freqs, global_word_to_idx = compute_word_freqs(fname)

	print(len(global_word_freqs))

	b = os.path.getsize(fname)

	B_size = b//_NTHR+1

	#~ parameters = [{"fname" : fname, "idx_to_seek" : x, "blocksize" : B_size, "window": _WINSIZE, "freqdist": global_word_freqs, "fr_threshold": _MINCOUNT, "word_to_idx_map": global_word_to_idx} for x in range(0, b, B_size)]
	parameters = [{"fname" : fname, "window": _WINSIZE, "freqdist": global_word_freqs, "fr_threshold": _MINCOUNT, "word_to_idx_map": global_word_to_idx} for x in range(0, b, B_size)]


	#~ p = multiprocessing.Pool(_NTHR)

	print(datetime.now())
	print("before mapping")

	#~ res = [p.apply_async(count_cooc, kwds=pardict) for pardict in parameters]
	#~ out = [r.get() for r in res]


	data, rows, columns, new_map = count_cooc(**parameters[0])
	n_dim = len(new_map)
	mat = sparse.csr_matrix((data, (rows, columns)), shape=(n_dim, n_dim), dtype='f')

	idx_to_word_map = {}

	for w, k in global_word_to_idx.items():
		if k in new_map:
			idx_to_word_map[k] = w

	print("SHAPE")
	print(mat.shape)
	print("NONZERO")
	print(mat.getnnz())

	sparse.save_npz("{}.mincount-{}".format(fname,_MINCOUNT), mat)

	with open("{}.mincount-{}.map".format(fname, _MINCOUNT), "w") as fout:
		for k, w in idx_to_word_map.items():
			fout.write("{}\t{}\n".format(k, w))

	print(datetime.now())
	print("done dumping")
