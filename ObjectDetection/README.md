# Object Detection Module

There is no training performed in this module. FasterRCNN on Inception-ResNet V2 module has been used to generate bounding boxes.

Place the images you wish to perform Object detection in `Data/Imgs` folder if you would like to run on default parameter settings. 

In `Main.py`, set the parameters of `PLOT_TOP_K` to predict how many bounding boxes you would like to display and give the path of folder of images which you would like to detect objects in `INFER_PATH` parameter. Do not change the `BATCH_SIZE` parameter as currently batching is not supported.
    
To run the code,
  
    python3 Main.py

This is the sample output from this module.

![Object Detection example](../ReadMeImages/object_detection.jpg)
