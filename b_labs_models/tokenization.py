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

from b_labs_models.settings import TOKENIZATION_MODEL_PATH


def char2features(text, i):
    char = text[i]
    length = len(text)

    features = [
        'lower={0}'.format(char.lower()),
        'isupper={0}'.format(char.isupper()),
        'isnumeric={0}'.format(char.isnumeric()),
    ]

    if i == 0:
        features.extend(['BOS'])

    if i > 0:
        char = text[i - 1]
        features.extend([
            '-1:lower={0}'.format(char.lower()),
            '-1:isupper={0}'.format(char.isupper()),
            '-1:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i > 1:
        char = text[i - 2]
        features.extend([
            '-2:lower={0}'.format(char.lower()),
            '-2:isupper={0}'.format(char.isupper()),
            '-2:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i < length - 1:
        char = text[i + 1]
        features.extend([
            '+1:lower={0}'.format(char.lower()),
            '+1:isupper={0}'.format(char.isupper()),
            '+1:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i < length - 2:
        char = text[i + 2]
        features.extend([
            '+2:lower={0}'.format(char.lower()),
            '+2:isupper={0}'.format(char.isupper()),
            '+2:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i == length - 1:
        features.extend(['EOS'])

    return features


def text2labels(text, words):
    for word in words:
        length = len(word)
        index = text.index(word)
        replacement = 'B' + ('I' * (length - 1))
        text = '{prefix}{replacement}{suffix}'.format(
            prefix=text[:index],
            replacement=replacement,
            suffix=text[index + length:]
        )
    labels = [c for c in text]
    for i, char in enumerate(labels):
        if char not in {'B', 'I'}:
            labels[i] = 'O'
    return labels


def text2features(text):
    return (
        char2features(text, i) for i, _ in enumerate(text)
    )


def labels2tokens(text, labels):
    buff = ''
    for i, c in enumerate(labels):
        if c == 'B':
            if buff:
                yield buff
            buff = text[i]
        elif c == 'I':
            buff += text[i]
        elif c == 'O':
            if buff:
                yield buff
            buff = ''
    if buff:
        yield buff


def get_train_data(corpus, count=None, **kwargs):
    X = []
    y = []

    documents = corpus.iter_documents()
    if count:
        documents = islice(documents, count)

    for document in tqdm(documents):
        try:
            text = document.raw()
            words = document.words()

            labels = text2labels(text, words)
            features = list(text2features(text))

            X.append(features)
            y.append(labels)
        except Exception as exc:
            # TODO:
            continue

    return train_test_split(X, y, **kwargs)


def train(X_train, y_train, **kwargs):
    '''
    >>> corpus = CorpusReader('annot.opcorpora.xml')
    >>> X_train, x_test, y_train, y_test = get_train_data(corpus, test_size=0.33, random_state=42)
    >>> crf = train(X_train, y_train)
    '''
    crf = Trainer()
    crf.set_params({
        'c1': 1.0,
        'c2': 0.001,
        'max_iterations': 200,
        'feature.possible_transitions': True,
    })

    for xseq, yseq in zip(X_train, y_train):
        crf.append(xseq, yseq)
    crf.train(TOKENIZATION_MODEL_PATH)
    return crf


class Tokenizer:

    '''
    Simple interface to CRFSuite tagger, that returns labels for
    each char in given text (actually, does tokenizing)
    '''

    def __init__(self, tagger=None):
        if not tagger:
            tagger = Tagger()
            tagger.open(TOKENIZATION_MODEL_PATH)
        self.tagger = tagger

    def split(self, sentence):
        labels = self.tagger.tag(text2features(sentence))
        return labels2tokens(sentence, labels)
