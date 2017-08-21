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

from b_labs_models.settings import PART_OF_SPEECH_MODEL_PATH


def token2posfeatures(sentence, i):
    token = sentence[i]
    length = len(sentence)

    features = [
        'lower={0}'.format(token.lower()),
        'isupper={0}'.format(token.isupper()),
        'istitle={0}'.format(token.istitle()),
        'isdigit={0}'.format(token.isdigit()),
        'prefix={0}'.format(token[:3].lower()),
        'suffix={0}'.format(token[-3:].lower()),
        'content={0}'.format(token[3:-3].lower()),
    ]

    if i == 0:
        features.extend(['BOS'])

    if i > 0:
        char = sentence[i - 1]
        features.extend([
            '-1:lower={0}'.format(token.lower()),
            '-1:isupper={0}'.format(token.isupper()),
            '-1:istitle={0}'.format(token.istitle()),
            '-1:isdigit={0}'.format(token.isdigit()),
            '-1:prefix={0}'.format(token[:3].lower()),
            '-1:suffix={0}'.format(token[-3:].lower()),
            '-1:content={0}'.format(token[3:-3].lower()),
        ])

    if i > 1:
        char = sentence[i - 2]
        features.extend([
            '-2:lower={0}'.format(token.lower()),
            '-2:isupper={0}'.format(token.isupper()),
            '-2:istitle={0}'.format(token.istitle()),
            '-2:isdigit={0}'.format(token.isdigit()),
            '-2:prefix={0}'.format(token[:3].lower()),
            '-2:suffix={0}'.format(token[-3:].lower()),
            '-2:content={0}'.format(token[3:-3].lower()),
        ])
    if i > 2:
        char = sentence[i - 3]
        features.extend([
            '-3:lower={0}'.format(token.lower()),
            '-3:isupper={0}'.format(token.isupper()),
            '-3:istitle={0}'.format(token.istitle()),
            '-3:isdigit={0}'.format(token.isdigit()),
            '-3:prefix={0}'.format(token[:3].lower()),
            '-3:suffix={0}'.format(token[-3:].lower()),
            '-3:content={0}'.format(token[3:-3].lower()),
        ])

    if i > 3:
        char = sentence[i - 4]
        features.extend([
            '-4:lower={0}'.format(token.lower()),
            '-4:isupper={0}'.format(token.isupper()),
            '-4:istitle={0}'.format(token.istitle()),
            '-4:isdigit={0}'.format(token.isdigit()),
            '-4:prefix={0}'.format(token[:3].lower()),
            '-4:suffix={0}'.format(token[-3:].lower()),
            '-4:content={0}'.format(token[3:-3].lower()),
        ])

    if i > 4:
        char = sentence[i - 5]
        features.extend([
            '-5:lower={0}'.format(token.lower()),
            '-5:isupper={0}'.format(token.isupper()),
            '-5:istitle={0}'.format(token.istitle()),
            '-5:isdigit={0}'.format(token.isdigit()),
            '-5:prefix={0}'.format(token[:3].lower()),
            '-5:suffix={0}'.format(token[-3:].lower()),
            '-5:content={0}'.format(token[3:-3].lower()),
        ])

    if i < length - 1:
        char = sentence[i + 1]
        features.extend([
            '+1:lower={0}'.format(token.lower()),
            '+1:isupper={0}'.format(token.isupper()),
            '+1:istitle={0}'.format(token.istitle()),
            '+1:isdigit={0}'.format(token.isdigit()),
            '+1:prefix={0}'.format(token[:3].lower()),
            '+1:suffix={0}'.format(token[-3:].lower()),
            '+1:content={0}'.format(token[3:-3].lower()),
        ])

    if i < length - 2:
        char = sentence[i + 2]
        features.extend([
            '+2:lower={0}'.format(token.lower()),
            '+2:isupper={0}'.format(token.isupper()),
            '+2:istitle={0}'.format(token.istitle()),
            '+2:isdigit={0}'.format(token.isdigit()),
            '+2:prefix={0}'.format(token[:3].lower()),
            '+2:suffix={0}'.format(token[-3:].lower()),
            '+2:content={0}'.format(token[3:-3].lower()),
        ])

    if i < length - 3:
        char = sentence[i + 3]
        features.extend([
            '+3:lower={0}'.format(token.lower()),
            '+3:isupper={0}'.format(token.isupper()),
            '+3:istitle={0}'.format(token.istitle()),
            '+3:isdigit={0}'.format(token.isdigit()),
            '+3:prefix={0}'.format(token[:3].lower()),
            '+3:suffix={0}'.format(token[-3:].lower()),
            '+3:content={0}'.format(token[3:-3].lower()),
        ])

    if i < length - 4:
        char = sentence[i + 4]
        features.extend([
            '+4:lower={0}'.format(token.lower()),
            '+4:isupper={0}'.format(token.isupper()),
            '+4:istitle={0}'.format(token.istitle()),
            '+4:isdigit={0}'.format(token.isdigit()),
            '+4:prefix={0}'.format(token[:3].lower()),
            '+4:suffix={0}'.format(token[-3:].lower()),
            '+4:content={0}'.format(token[3:-3].lower()),
        ])

    if i < length - 5:
        char = sentence[i + 5]
        features.extend([
            '+5:lower={0}'.format(token.lower()),
            '+5:isupper={0}'.format(token.isupper()),
            '+5:istitle={0}'.format(token.istitle()),
            '+5:isdigit={0}'.format(token.isdigit()),
            '+5:prefix={0}'.format(token[:3].lower()),
            '+5:suffix={0}'.format(token[-3:].lower()),
            '+5:content={0}'.format(token[3:-3].lower()),
        ])

    if i == length - 1:
        features.extend(['EOS'])

    return features


def sent2posfeatures(sent):
    return [
        token2posfeatures(sent, i) for i, _ in enumerate(sent)
    ]


def get_pos_train_data(corpus, count=None, **kwargs):
    X = []
    y = []

    documents = corpus.iter_documents()
    if count:
        documents = islice(documents, count)

    for document in tqdm(documents):
        sents = document.iter_tagged_sents()
        for sent in sents:
            tokens = []
            labels = []
            for token, tags in sent:
                tags = tags.split(',')
                tokens.append(token)
                labels.append(tags[0])  # TODO:
            X.append(sent2posfeatures(tokens))
            y.append(labels)

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
    crf.train(PART_OF_SPEECH_MODEL_PATH)
    return crf


class POSTagger:

    def __init__(self, tagger=None):
        if not tagger:
            tagger = Tagger()
            tagger.open(PART_OF_SPEECH_MODEL_PATH)
        self.tagger = tagger

    def tag(self, tokens):
        return self.tagger.tag(sent2posfeatures(tokens))
