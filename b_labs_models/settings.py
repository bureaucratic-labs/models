import os


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODELS_PATH = os.path.join(BASE_PATH, 'data')

SENTENCE_SEGMENTATION_MODEL_PATH = os.path.join(
    MODELS_PATH, 'sentence-segmentation-model.crfsuite',
)

TOKENIZATION_MODEL_PATH = os.path.join(
    MODELS_PATH, 'tokenization-model.crfsuite',
)

PART_OF_SPEECH_MODEL_PATH = os.path.join(
    MODELS_PATH, 'part-of-speech.crfsuite',
)
