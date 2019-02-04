"""Generate three different sets of sentences, derived from coinco datasets."""
import os
import logging
import gzip
import collections
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

__all__ = ('generate_sentences_datasets')


def _load_vocabulary(map_filepath):
    voc = set()

    with open(map_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            voc.add(line.strip().split('\t')[1].lower())
    return voc


def generate(output_dirpath, coinco_filepath, space_filepath):
    basename_space_filepath = os.path.basename(space_filepath)
    map_filepath = '{}.map'.format(basename_space_filepath)
    if basename_space_filepath.endswith('.npz'):
        map_filepath = '{}.map'.format(basename_space_filepath[:-len('.npz')])

    vocabulary = _load_vocabulary(map_filepath)

    coinco = ET.fromstring(gzip.open(coinco_filepath, "rt").read())
    sentences = collections.defaultdict(list)
    for sentence in coinco:
        keep_sentence = True
#        target = sentence.findall("targetsentence")[0].text.strip()
        tokens = sentence.findall("tokens")[0].findall("token")
        complete_tokens = []
        content_tokens_number = 0.0
        for tok in tokens:
            content_token = True
            wordform = tok.attrib["wordform"].lower()

            pos = tok.attrib["posTT"]
            subst_list = []
            if len(tok.findall("substitutions")):
                substs = tok.findall("substitutions")[0].findall("subst")
                subst_lemmas = [s.attrib["lemma"].lower() for s in substs]
                subst_list.extend([s for s in subst_lemmas if len(s.split()) == 1
                                   and s in vocabulary])
            if not (len(pos) == 1 or pos == 'NP' or all(not c.isalpha()
                                                        for c in wordform)):
                content_token = False
                wordform = wordform+'/N'
            elif wordform not in vocabulary:
                keep_sentence = False
            subst_list = [wordform]+subst_list
            if content_token:
                content_tokens_number += 1
            complete_tokens.append(subst_list)
    content_ratio = content_tokens_number/len(complete_tokens)
    if content_tokens_number < 4 and content_ratio < 0.79:
        keep_sentence = False
    if keep_sentence:
        sentences[content_tokens_number].append(complete_tokens)
