from Train import Train

params = {
    'EPOCHS': 15,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.0001,                    # Starting Learning rate
    'N_CLASS': 85,
    'DIVIDE_LEARNING_RATE_AT': [5],          # Which epochs, learning rate should be divided. Starts from 0.
    'IMAGE_SIZE': (224, 224),

    'TRAIN_PATH': './Data/train',
    'TEST_PATH': './Data/test',
    'TRAIN_VAL_RATIO': 0.98,       # Keeping a small percent for validation data
                                   # We are not doing stratified sampling
    'DATA_LABELS': './Data/meta-data/train.csv'
}
t = Train(params)
t.train()
t.test()
