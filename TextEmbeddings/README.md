# Text Embedding Module
Text embeddings can be performed at individual word level present inside the sentence or can be performed at the entire sentence level. In this example, both types of embeddings i.e at word level and at sentence level has been used on sentiment analysis problem. In both the approaches, Elmo embeddings have been made use of.

Download the data from [Kaggle's movie review sentiment analysis](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) problem and extract the contents to `SentenceLevel/Data` folder and `WordLevel/Data` folder if you would like to run using default parameter settings.

In `SentenceLevel/Main.py` and `WordLevel/Main.py`, these are the explanations of various parameters present:

`EPOCHS`: Number of epochs you would like to run the code

`BATCH_SIZE`: Batch size during training

`LEARNING_RATE`: Learning rate during training

`TRAIN_FILE`: The file path where your train data file is present.

`TEST_FILE`: The file path where your test data file is present.

`TRAIN_VAL_RATIO`: The train-validation ratio to be performed on your train dataset.

`N_CLASS`: The number of classes present in the dataset

In addition to above mentioned parameters, the `WordLevel/Main.py` has extra parameters which are explained below:

`DIVIDE_LEARNING_RATE_AT`: At which epochs, learning rate should be divided by 10. Epoch count starts from 0.

`MAX_SENTENCE_WORDS`: How many words in a sentence is to be considered.

To run the text embedding at word level code:

    cd WordLevel
    python3 Main.py

To run the text embedding at sentence level code,

    cd SentenceLevel
    python3 Main.py

