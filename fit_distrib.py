import numpy as np

def generated_biased_dims(svd_dims, bias_dims, block_size):
    """Generate a set of dimensions, biased by input bias_dims.

    Return the same number of dims as bias_dims, sampled from svd_dims.
    """
    blocks = []
    count = 0
    for idx in svd_dims:
        if idx in bias_dims:
            count += 1
        if idx % block_size == 0 and idx != 0:
            blocks.append(count)
            count = 0
    total = sum(x for x in blocks)
    block_probs = [x / total for x in blocks]
    assert np.isclose(1, sum(x for x in block_probs))
    probs = []
    block_idx = 0
    for idx in svd_dims:
        if idx % block_size == 0 and idx != 0:
            block_idx += 1
        if block_idx < len(blocks):
            probs.append(block_probs[block_idx] / block_size)
        else:
            probs.append(0.)
    assert np.isclose(1, sum(x for x in probs))
    return sorted(np.random.choice(svd_dims, size=len(bias_dims), replace=False,
                                   p=probs))

if __name__ == '__main__':
    dims = [5, 10, 100, 500, 685, 899, 1500]
    print(generated_biased_dims(list(range(0, 10000)), dims, 30))
