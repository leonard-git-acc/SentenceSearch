# SentenceSearch
SentenceSearch is a project, that aims at question answering tasks. It aims at answering a question with a sentence instead of returning the correct passage, therefore it has been trained on a modified SQuAD dataset.

## What is possible?
When testing the demo I found some interesting examples, which demonstrate what SentenceSearch can do:

### Question1:
    Where was Pierre on a Friday evening?
### Question2:
    Where was Pierre on the last work day late afternoon?

### Answer of SentenceSearch to both Questions:
    One Friday evening, Pierre was down in the winery, working on a new electric winepress.

## Setup
To run SentenceSearch and its demo it is necessary to install some packages:

- pip install tensorflow-gpu (if you want to use the cpu version change the CuDNNGRU's to GRU's)
- pip install tensorflow-hub
- pip install numpy
- pip install nltk
    
    nltk needs some additional setup:
    - open python and run "import nltk" and "nltk.download", which opens a window
    - in collections download popular packages

To start training you need to download the SQuAD v1.1 dataset and parse them with squad_parse.py. The resulting .json files are the train and test files for train.py.
For faster training have a look at the squad_vectorize.py.


