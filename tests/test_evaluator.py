"""Test the evaluator."""

import entropix.core.evaluator as evaluator


def test_load_kfold_splits_dict_nodev():
    left_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    right_idx = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    sim = [1, 0, .5, 1, 0, .5, .5, 1, 0, 1]
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.2, dev_type='nodev')
    assert len(kfold_dict.keys()) == 5
    assert len(kfold_dict[1]['train'].keys()) == 3
    assert len(kfold_dict[1]['train']['sim']) == 8
    assert len(kfold_dict[1]['test']['sim']) == 2
    assert kfold_dict[1]['test']['left_idx'] == [0, 1]
    assert kfold_dict[1]['test']['right_idx'] == [9, 8]
    assert kfold_dict[4]['test']['left_idx'] == [6, 7]
    assert kfold_dict[4]['test']['right_idx'] == [3, 2]
    assert kfold_dict[4]['train']['left_idx'] == [0, 1, 2, 3, 4, 5, 8, 9]
    assert kfold_dict[4]['train']['right_idx'] == [9, 8, 7, 6, 5, 4, 1, 0]
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.05, dev_type='nodev')
    assert len(kfold_dict.keys()) == 10
    assert len(kfold_dict[1]['test']['sim']) == 1
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.3, dev_type='nodev')
    assert len(kfold_dict.keys()) == 3
    assert len(kfold_dict[1]['test']['sim']) == 3
    assert len(kfold_dict[3]['test']['sim']) == 3
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.4, dev_type='nodev')
    assert len(kfold_dict.keys()) == 2
    assert len(kfold_dict[1]['test']['sim']) == 4
    assert len(kfold_dict[2]['test']['sim']) == 4
    assert kfold_dict[2]['test']['right_idx'] == [5, 4, 3, 2]

    left_idx = list(range(1, 999))
    right_idx = list(range(1, 999))
    sim = list(range(1, 999))
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.2, dev_type='nodev')
    assert len(kfold_dict.keys()) == 5


def test_load_kfold_splits_dict_regular():
    left_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    right_idx = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    sim = [1, 0, .5, 1, 0, .5, .5, 1, 0, 1]
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.2, dev_type='regular')
    assert len(kfold_dict.keys()) == 5
    assert len(kfold_dict[1]['train'].keys()) == 3
    assert len(kfold_dict[1]['train']['sim']) == 6
    assert len(kfold_dict[1]['dev']['sim']) == 2
    assert len(kfold_dict[1]['test']['sim']) == 2
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.05, dev_type='regular')
    assert len(kfold_dict.keys()) == 10
    assert len(kfold_dict[1]['test']['sim']) == 1
    assert len(kfold_dict[1]['dev']['sim']) == 1
    assert len(kfold_dict[1]['train']['sim']) == 8
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.3, dev_type='regular')
    assert len(kfold_dict.keys()) == 3
    assert len(kfold_dict[1]['test']['sim']) == 3
    assert len(kfold_dict[1]['dev']['sim']) == 3
    assert len(kfold_dict[1]['train']['sim']) == 4
    kfold_dict = evaluator._load_kfold_splits_dict(
        left_idx, right_idx, sim, kfold_size=.4, dev_type='regular')
    assert len(kfold_dict[1]['test']['sim']) == 4
    assert len(kfold_dict[1]['dev']['sim']) == 4
    assert len(kfold_dict[1]['train']['sim']) == 2
