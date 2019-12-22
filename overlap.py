if __name__ == '__main__':
    CORPUS1 = '/Users/akb/Github/entropix/corpora/enwiki2.txt'
    CORPUS2 = '/Users/akb/Github/entropix/corpora/enwiki4.txt'
    counter = 0
    total_sent = 0
    sent2s = set()
    with open(CORPUS2, 'r', encoding='utf-8') as stream2:
        for sent in stream2:
            sent2s.add(sent)
    with open(CORPUS1, 'r', encoding='utf-8') as stream1:
        for sent in stream1:
            total_sent += 1
            if sent in sent2s:
                counter += 1
    print('Number of sent of CORPUS1 in CORPUS2 = {}'.format(counter))
    overlap = (counter / total_sent) * 100
    print('Overlap = {}%'.format(overlap))
