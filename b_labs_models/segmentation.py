from itertools import islice
from pycrfsuite import Trainer, Tagger

try:
    # tagging and training support
    from tqdm import tqdm
    from opencorpora import CorpusReader
    from sklearn.model_selection import train_test_split
except ImportError:
    # only tagging support
    pass

from b_labs_models.settings import SENTENCE_SEGMENTATION_MODEL_PATH


def char2features(sentence, i):
    '''
    Returns features of char at position `i` in given sentence
    When it possible, result features list also includes features for 2 characters
    ahead and behind current position (like bigrams, or something like it)
    Currently, used features is:
    1. lower-cased value of character
    2. result of calling `character.isupper()` method
    3. result of calling `character.isnumeric()` method
    '''
    char = sentence[i]
    length = len(sentence)

    features = [
        'lower={0}'.format(char.lower()),
        'isupper={0}'.format(char.isupper()),
        'isnumeric={0}'.format(char.isnumeric()),
    ]

    if i == 0:
        features.extend(['BOS'])

    if i > 0:
        char = sentence[i - 1]
        features.extend([
            '-1:lower={0}'.format(char.lower()),
            '-1:isupper={0}'.format(char.isupper()),
            '-1:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i > 1:
        char = sentence[i - 2]
        features.extend([
            '-2:lower={0}'.format(char.lower()),
            '-2:isupper={0}'.format(char.isupper()),
            '-2:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i < length - 1:
        char = sentence[i + 1]
        features.extend([
            '+1:lower={0}'.format(char.lower()),
            '+1:isupper={0}'.format(char.isupper()),
            '+1:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i < length - 2:
        char = sentence[i + 2]
        features.extend([
            '+2:lower={0}'.format(char.lower()),
            '+2:isupper={0}'.format(char.isupper()),
            '+2:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i == length - 1:
        features.extend(['EOS'])

    return features


def text2labels(text, sents):
    '''
    Marks all characters in given `text`, that doesn't exists within any
    element of `sents` with `1` character, other characters (within sentences)
    will be marked with `0`
    Used in training process
    >>> text = 'привет. меня зовут аня.'
    >>> sents = ['привет.', 'меня зовут аня.']
    >>> labels = text2labels(text, sents)
    >>> ' '.join(text)
    >>> 'п р и в е т .   м е н я   з о в у т   а н я .'
    >>> ' '.join(labels)
    >>> '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    '''
    labels = [c for c in text]
    for sent in sents:
        start = text.index(sent)
        finish = start + len(sent)
        labels[start:finish] = '0' * len(sent)
    for i, c in enumerate(labels):
        if c != '0':
            labels[i] = '1'
    return labels


def sent2features(sent):
    '''
    Returns list with per-character features for given sentence
    '''
    return [
        char2features(sent, i) for i, _ in enumerate(sent)
    ]


def text2sentences(text, labels):
    '''
    Splits given text at predicted positions from `labels`
    '''
    sentence = ''
    for i, label in enumerate(labels):
        if label == '1':
            if sentence:
                yield sentence
            sentence = ''
        else:
            sentence += text[i]
    if sentence:
        yield sentence


def get_train_data(corpus, **kwargs):
    X = []
    y = []

    documents = corpus.iter_documents()

    if count:
        documents = islice(documents, count)

    for document in tqdm(documents):
        try:
            text = document.raw()
            sents = document.raw_sents()

            labels = text2labels(text, sents)
            features = sent2features(text)

            X.append(features)
            y.append(labels)
        except Exception as exc:
            # TODO:
            pass

    return train_test_split(X, y, **kwargs)


def train(X_train, X_test, y_train, y_test, **kwargs):
    '''
    >>> corpus = CorpusReader('annot.opcorpora.xml')
    >>> X_train, x_test, y_train, y_test = get_train_data(corpus, test_size=0.33, random_state=42)
    >>> crf = train(X_train, X_test, y_train, y_test)
    '''
    crf = Trainer()
    crf.set_params({
        'c1': 1.0,
        'c2': 0.001,
        'max_iterations': 200,
    })

    for xseq, yseq in zip(X_train, y_train):
        crf.append(xseq, yseq)
    crf.train(model_name)
    return crf


class SentenceSegmentator:

    def __init__(self, tagger=None):
        if not tagger:
            tagger = Tagger()
            tagger.open(SENTENCE_SEGMENTATION_MODEL_PATH)
        self.tagger = tagger

    def split(self, text):
        labels = self.tagger.tag(sent2features(text))
        return text2sentences(text, labels)
