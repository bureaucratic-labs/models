# Ready to use CRFSuite models for russian language

## Sentence segmentation

F1 score on cross-validation (1/3 of data is test data): 0.997  
Last update: 30 may 2017  
Train data: 4k+ OpenCorpora articles (mostly news and fiction literature)  

```bash
$ python sentence-segmentation.py 
Input text: Разница цепей Маркова от сетей Маркова заключается в том, что первые генеративны (т.е. предсказывают вероятность следующего шага), а вторые — дискриминатины, т.е. рассчитывают вероятность текущего состояния. Использовать тот или иной алгоритм зависит от решаемой задачи. А второе, и наиболее важное отличие — это то, что сети Маркова учитывают не только шаг (два и т.д.) вправо-влево по какому-либо из параметров, а по пучку взаимосвязанных параметров. Скажем, для перевода это не только все его варианты, а и тематический контекст перевода, синтаксис и пр.

# =>

0: Разница цепей Маркова от сетей Маркова заключается в том, что первые генеративны (т.е. предсказывают вероятность следующего шага), а вторые — дискриминатины, т.е. рассчитывают вероятность текущего состояния.
1: Использовать тот или иной алгоритм зависит от решаемой задачи.
2: А второе, и наиболее важное отличие — это то, что сети Маркова учитывают не только шаг (два и т.д.) вправо-влево по какому-либо из параметров, а по пучку взаимосвязанных параметров.
3: Скажем, для перевода это не только все его варианты, а и тематический контекст перевода, синтаксис и пр.
```

## Tokenization

F1 score on cross-validation (1/3 of data is test data): 0.98  
Last update: 3 june 2017  
Train data: 4k+ OpenCorpora articles (mostly news and fiction literature)  

```bash
$ python tokenization.py
python tokenization.py 
Input text: Плита дорожная железобетонная ПДН.м Серия 3.503.1-91, выпуск 1
П л и т а   д о р о ж н а я   ж е л е з о б е т о н н а я   П Д Н . м   С е р и я   3 . 5 0 3 . 1 - 9 1 ,   в ы п у с к   1
B I I I I O B I I I I I I I O B I I I I I I I I I I I I I O B I I B I O B I I I I O B I I I I I I I I I B O B I I I I I O B
```

## License

Source code licensed under MIT license, but source data (OpenCorpora annotated corpus, for example) may have different license.
