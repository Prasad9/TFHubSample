# Video Classification Module
This is a demonstration of video classification performed using Inception 3D module. 

Download the data from [20BN-Jester V1 dataset](https://20bn.com/datasets/jester). Follow the instructions in their website and extract them. Place the resulting video folder in `Data` folder if you would like to run on default parameter settings.

Inception 3D module has a necessity of fixed size input of 224 x 224. Hence you will have to resize the downloaded videos into that fixed size. Also, we will be fixing the number of frames in each video which is currently set at 36. In Preprocess.py, set the parameter of `SRC_PATH` to the folder path where the downloaded videos are present and `DST_PATH` to the folder path where the resized videos should be stored in.

Run the following code to resize the videos.

    python3 Preprocess.py

After resizing, set the parameters in `Main.py` which are explained next.

`EPOCHS`: Number of epochs you would like to run the code

`BATCH_SIZE`: Batch size during training

`LEARNING_RATE`: Starting Learning rate

`DIVIDE_LEARNING_RATE_AT`: At which epochs, learning rate should be divided by 10. Epoch count starts from 0.

`DATA_PATH`: Folder path of the resized videos folder

`TRAIN_LABELS`: File path of the downloaded train file

`VAL_LABELS`: File path of the downloaded validation file

`TEST_LABELS`: File path of the downloaded test file

`LABEL_TEXT`: File path of the downloaded labels file

`SEQ_LEN`: Number of frames in each video clip

`IMG_WIDTH`: Width of video

`IMG_HEIGHT`: Height of video

`REQD_LABELS`: The set of labels you wish to train on.

To start the training, run the following code:

    python3 Main.py

After the training, to test the model directly using your camera, configure the parameters in `Infer.py` which is explained below:

`SEQ_LEN`: Number of frames in each video clip

`IMG_WIDTH`: Width of video

`IMG_HEIGHT`: Height of video

`LABELS`: List all the labels you have trained your model on. Do not change the order from that used during training

`CHECKPOINT_PATH`: Folder path where your saved_model is present

To start the real-time testing application, run the following code

    python3 Infer.py

This is the sample output result of performing inference on the trained module.

![Video Classification demo](../ReadMeImages/video_classification.gif)