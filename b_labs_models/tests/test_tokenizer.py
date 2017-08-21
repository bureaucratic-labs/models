import pytest


from b_labs_models import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


def test_split_tokens(tokenizer):
    tokens = list(tokenizer.split('тест один два три.'))
    assert tokens == ['тест', 'один', 'два', 'три', '.']


def test_split_more_complex_tokens(tokenizer):
    tokens = list(tokenizer.split('Это МиГ-17 на хвосту ...'))
    assert tokens == ['Это', 'МиГ-17', 'на', 'хвосту', '...']
