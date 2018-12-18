# Image Augmentation Module
Training is not possible on Image Augmentation modules as they contain no variables.

## Direct Images
This is a demonstration of making use of Image Augmentation module performing directly on decoded image data. The AutoAugment algorithm Module has been used in this example. 

In `DirectImages/Main.py`, set the parameter of `OUTPUT_IMAGE_SIZE` to the output shape of your image, `AUGMENTATIONS_PER_IMAGE` to number of augmented images you would like to generate per input image. `SRC_PATH` refers to the folder of images where your input images are present and `DST_PATH` refers to the folder where your augmented images should be stored in.

To run the code,

    cd DirectImages
    python3 Main.py

This is how the likely augmented images will look like:

![Augmentation Example](../ReadMeImages/image_augmentation.jpg)

## Clubbed Modules
This is a demonstration of making use of Image Augmentation module paired with other TF-Hub modules. Also, it demonstrates how to perform augmentations on encoded images.The Crop-Color Image Augmentation Module has been used in this example. The images generated from this module is passed into Resnet 50 V2 Feature Vector module which will be trained on multi-label classification problem.

Download the data from [Hackerearth's predicting attribute of animal](https://www.hackerearth.com/problem/machine-learning/predict-the-energy-used-612632a9/) problem. Extract it and place the contents inside `Data` folder to run with default parameters. 

In `ClubbedModules/Main.py`, these are the explanations of the parameters present:

`EPOCHS`: Number of epochs you would like to run the code

`BATCH_SIZE`: Batch size during training

`LEARNING_RATE`: Starting Learning rate

`N_CLASS`: Number of classes present in dataset

`DIVIDE_LEARNING_RATE_AT`: At which epochs, learning rate should be divided by 10. Epoch count starts from 0.

`IMAGE_SIZE`: The image size to be passed to Feature Vector module as input.

`TRAIN_PATH`: The folder path where your train data folder is present.

`TEST_PATH`: The folder path where your test data folder is present.

`TRAIN_VAL_RATIO`: The train-validation ratio to be performed on your train dataset

`DATA_LABELS`: The file path where your labels file is present.

To run the code,

    cd ClubbedModules
    python3 Main.py