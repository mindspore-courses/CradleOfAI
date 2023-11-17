"""
author: Ruben Tao
"""

from typing import List, Dict, Union


def _get_label_dictionary():
    dictionary = {'O': 0, 'B-Symptom': 1, 'I-Symptom': 2, 'B-Drug': 3,
                  'I-Drug': 4, 'B-Drug_Category': 5, 'I-Drug_Category': 6, 'B-Medical_Examination': 7,
                  'I-Medical_Examination': 8, 'B-Operation': 9, 'I-Operation': 10, '<pad>': 11}
    return dictionary


def _get_label_vocab():
    vocab = ['O', 'B-Symptom', 'I-Symptom', 'B-Drug',
             'I-Drug', 'B-Drug_Category', 'I-Drug_Category', 'B-Medical_Examination',
             'I-Medical_Examination', 'B-Operation', 'I-Operation', '<pad>']
    return vocab


def tokenize(src: Union[str, List[str]], dictionary: Dict = None, length: int = 0):
    res = []
    if dictionary is None:
        dictionary = _get_label_dictionary()
    for val in src:
        try:
            res.append(dictionary[val])
        except KeyError:
            res.append(dictionary['<unk>'])
    if length > 0:
        while len(res) < length:
            res.append(dictionary['<pad>'])
    return res


def untokenize(tokens: List[int], vocab: List[str] = None, add_space: bool = True):
    sentence = ''
    if vocab is None:
        vocab = _get_label_vocab()
    for token in tokens:
        sentence += vocab[token]
        if add_space:
            sentence += ' '
    return sentence.rstrip()
