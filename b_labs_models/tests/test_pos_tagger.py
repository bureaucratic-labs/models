import pytest

from b_labs_models import POSTagger


@pytest.fixture
def tagger():
    return POSTagger()


def test_tag_tokens(tagger):
    tokens = ['Весело', 'стучали', 'храбрые', 'сердца']
    labels = tagger.tag(tokens)
    assert list(zip(tokens, labels)) == [
        ('Весело', 'ADJS'),
        ('стучали', 'VERB'),
        ('храбрые', 'ADJF'),
        ('сердца', 'NOUN'),
    ]
