# Feature Vector Module
This is a demonstration of fine-tuning the ResNet 50 V2 feature vector module to predict in a multi-label classification problem. The final layers of the model has been fine-tuned along with the customly appended dense layers.

Download the data from [Hackerearth's predicting attribute of animal](https://www.hackerearth.com/problem/machine-learning/predict-the-energy-used-612632a9/), and place the extracted contents from it in `Data` folder if you would like to work with default parameter settings. 

ResNet V2 module has the necessity of having the input image size of 224 x 224. Hence, we will have to preprocess the input images to the fixed size of 224 x 224. In `Preprocess.py` file, pass the downloaded train and test folder paths in `resize_paths` variable. Run the below code to start with the preprocessing:

    python3 Preprocess.py

To begin with the training procedure, configure the below parameters in `Main.py` file.

`EPOCHS`: Number of epochs you would like to run the code

`BATCH_SIZE`: Batch size during training

`LEARNING_RATE`: Starting Learning rate

`N_CLASS`: The number of classes present in the dataset

`DIVIDE_LEARNING_RATE_AT`: At which epochs, learning rate should be divided by 10. Epoch count starts from 0.

`TRAIN_PATH`: Folder path where your train folder has been resized to.

`TEST_PATH`: Folder path where your test folder has been resized to.

`TRAIN_VAL_RATIO`: The train-validation ratio to be performed on your train dataset.

`DATA_LABELS`: File path where the labels of train folder are present.

To start the training, run the below code:

    python3 Main.py