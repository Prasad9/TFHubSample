from Train import Train

params = {
    'EPOCHS': 6,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.0001,

    'TRAIN_FILE': './Data/train.tsv',
    'TEST_FILE': './Data/test.tsv',
    'TRAIN_VAL_RATIO': 0.95,

    'N_CLASS': 5
}

t = Train(params)
t.train()
t.test()
