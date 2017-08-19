from setuptools import setup, find_packages


setup(
    name='b-labs-models',
    version='2017.8.19',
    description='Ready to use CRFSuite models for sentence segmentation and tokenization',
    url='https://github.com/bureaucratic-labs/models',
    author='Dmitry Veselov',
    author_email='d.a.veselov@yandex.ru',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: Russian',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    packages=find_packages(),
    package_data={
        'b_labs_models': ['data/*'],
    },
    install_requires=['python-crfsuite'],
    extras_require={
        'train': [
            'tqdm',
            'numpy',
            'scipy',
            'scikit-learn',
            'opencorpora-tools',
        ],
        'test': [
            'pytest',
        ],
    }
)
