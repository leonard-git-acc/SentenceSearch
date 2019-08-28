# Pretrained Models
This folder contains on the modified SQuAD dataset pretrained models. There are two categories: gru and lstm models. Both perform differently on the Testset and with text_search.

## GRU Model Performance

### 10 training Epochs on Testset:
    Loss: 0.4742274988549096
    Accuracy: 0.7723214

### 50 training Epochs on Testset:
    Loss: 0.5374323331218746
    Accuracy: 0.78089285

### Note:
    GRU Models seem to produce more reasonable answers in text_search.

## LSTM Model Performance

### 10 training Epochs on Testset:
    Loss: 0.3529156221555812
    Accuracy: 0.8345536

### 50 training Epochs on Testset:
    Loss: 0.44802683145872185
    Accuracy: 0.81133926

### Note:
    LSTM Models seem to produce less reasonable answers in text_search, but the 50 is worse than the 10 training Epochs Model.


