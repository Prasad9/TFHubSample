from Train import Train

params = {
    'EPOCHS': 3,
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 0.0001,
    'DIVIDE_LEARNING_RATE_AT': [1, 2],      # At what epochs, learning rate should be divided. Starts at 0.

    'DATA_PATH': './Data/VideosResized224',
    'TRAIN_LABELS': './Data/train.csv',
    'VAL_LABELS': './Data/validation.csv',
    'TEST_LABELS': './Data/test.csv',
    'LABEL_TEXT': './Data/labels.csv',

    'SEQ_LEN': 36,                      # Number of frames in each video clip
    'IMG_WIDTH': 224,                   # Width of video
    'IMG_HEIGHT': 224,                  # Height of video

    # List all the labels you wish to train on.
    'REQD_LABELS': ['Swiping Left', 'Swiping Right', 'Swiping Down',
                    'Swiping Up', 'Pushing Hand Away', 'Pulling Hand In',
                    'Sliding Two Fingers Left', 'Sliding Two Fingers Right',
                    'Sliding Two Fingers Down', 'Sliding Two Fingers Up',
                    'Pushing Two Fingers Away', 'Pulling Two Fingers In',
                    'Rolling Hand Forward', 'Rolling Hand Backward', 'Turning Hand Clockwise',
                    'Turning Hand Counterclockwise', 'Zooming In With Full Hand',
                    'Zooming Out With Full Hand', 'Zooming In With Two Fingers',
                    'Zooming Out With Two Fingers', 'Thumb Up', 'Thumb Down',
                    'Shaking Hand', 'Stop Sign', 'Drumming Fingers',
                    'No gesture', 'Doing other things'],
}

t = Train(params)
t.train()
t.test()
