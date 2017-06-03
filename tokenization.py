import tqdm
import itertools


from opencorpora import CorpusReader
from pycrfsuite import Trainer, Tagger


from sklearn.model_selection import train_test_split


corpus = CorpusReader('annot.opcorpora.xml')


def char2features(text, i):
    char = text[i]

    features = [
        'lower={0}'.format(char.lower()),
        'isupper={0}'.format(char.isupper()),
        'isnumeric={0}'.format(char.isnumeric()),
    ]

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

    if i < len(text) - 1:
        char = text[i + 1]
        features.extend([
            '+1:lower={0}'.format(char.lower()),
            '+1:isupper={0}'.format(char.isupper()),
            '+1:isnumeric={0}'.format(char.isnumeric()),
        ])

    if i < len(text) - 2:
        char = text[i + 2]
        features.extend([
            '+2:lower={0}'.format(char.lower()),
            '+2:isupper={0}'.format(char.isupper()),
            '+2:isnumeric={0}'.format(char.isnumeric()),
        ])

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
        if char not in {'B', 'I', 'E'}:
            labels[i] = 'O'
    return labels


def text2features(text):
    return [
        char2features(text, i) for i, _ in enumerate(text)
    ]


def get_train_data(corpus, **kwargs):
    X = []
    y = []

    documents = corpus.iter_documents()

    for document in tqdm.tqdm(documents):
        try:
            text = document.raw()
            words = document.words()

            labels = text2labels(text, words)
            features = text2features(text)

            X.append(features)
            y.append(labels)
        except:
            continue

    return train_test_split(X, y, **kwargs)


model_name = 'data/tokenization-model.crfsuite'

X_train, X_test, y_train, y_test = get_train_data(corpus, test_size=0.10, random_state=42)

print('Train data:', len(X_train))
print('Test data:', len(X_test))

def train(X_train, y_train, **kwargs):
    '''
    >>> corpus = CorpusReader('annot.opcorpora.xml')
    >>> X_train, x_test, y_train, y_test = get_train_data(corpus, test_size=0.33, random_state=42)
    >>> crf = train(X_train, y_train,)
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
    crf.train(model_name)
    return crf

crf = train(X_train, y_train)


tagger = Tagger()
tagger.open(model_name)


while True:
    text = input('Input text: ')
    features = text2features(text)
    labels = tagger.tag(features)
    print(' '.join(text))
    print(' '.join(labels))
