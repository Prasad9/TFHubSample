from Train import Train

params = {
    'EPOCHS': 4,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.0001,

    'TRAIN_FILE': './Data/train.tsv',
    'TEST_FILE': './Data/test.tsv',
    'TRAIN_VAL_RATIO': 0.95,
    'MAX_SENTENCE_WORDS': 25,   # How many words in sentence is to be considered.

    'N_CLASS': 5
}

t = Train(params)
#t.train()
t.test()