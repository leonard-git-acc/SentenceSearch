# Pretrained Models
This folder contains on the modified SQuAD dataset pretrained models. There are two categories: gru and lstm models. Both perform differently on the testset and with text_search.

## Test1:
The first testing method is testing directly on the modified SQuAD testset. The newtork only has to determine, if a sentence matches a question or if it doesn't. Its metrics are accuracy (how often it matched the label correctly) and loss (how wrong it was with the prediction).

## Test2:
The second testing Method is also made on this testset, but in a broader sense. It is given a paragraph of 16 sentences and has to determine 2, that match the question the closest. This testing is similar to the demo and therefore tells the most about real performance. Its metrics are total (how many questions were asked), correct (correctly answered questions) and accuracy (correct / total)

## GRU Model Performance

### 10 Training Epochs:
    Test1: Loss=0.4742274988549096 Accuracy=0.7723214
    Test2: Correct=7427 Total=10570 Accuracy=0.702649006622516

### 20 Training Epochs
    Test1: Loss=0.4590620211937598 Accuracy=0.7974554
    Test2: Correct=7615 Total=10570 Accuracy=0.7204351939451277

### 50 Training Epochs:
    Test1: Loss=0.5374323331218746 Accuracy=0.78089285
    Test2: Correct=7392 Total=10570 Accuracy=0.6993377483443709

### Note:
    GRU Models seem to produce more reasonable answers in text_search.

## LSTM Model Performance

### 10 Training Epochs:
    Test1: Loss=0.3529156221555812 Accuracy=0.8345536
    Test2: Correct=7198 Total=10570 Accuracy=0.6809839167455062

### 50 Training Epochs:
    Test1: Loss=0.44802683145872185 Accuracy=0.81133926
    Test2: Correct=7177 Total=10570 Accuracy=0.6789971617786187

### Note:
    LSTM Models seem to produce less reasonable answers in text_search, but the 50 is worse than the 10 training epochs model.


