# Fracture.v1i_Reduced_SSD

From a selection of data from the Roboflow file https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1, which represents a reduced but homogeneous version of that file, try to perform fracture detection using SSD


By using only 147 images, training is allowed using a personal computer without GPU

===
Installation:

The packages used are the usual ones in a development environment with python. In case a module not found error appears, it will be enough to install it with a simple pip

To download the dataset you have to register as a roboflow user, but to simplify it, the compressed files for train, valid and test are attached: trainFractureOJumbo1.zip, validFractureOJumbo1.zip and testFractureOJumbo1.zip obtained by selecting the images that start with names 0 -_Jumbo-1 of the original file obtaining a reduced number of images that allow training with  a personal computer without GPU

Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

===
Training:

execute 

Train_Fracture.v1i_Reduced_SSD.py 

A log with its execution in 3000 epochs is attached: LOG_SSD_3000epoch.txt,

every 1000 epoch a model is written to the project directory with the name SSD_epochNNNN.pth where NNNN is the epoch number. The best epoch would be the corresponding to 3000 epoch

Due to their large size, over 1.3 Gb, it has not been possible to upload some verification models to github



TESTyolovFromScratch_kaggle.py modifying the name of the model that appears in instruction 3 according to the epoch to be considered.

The model that gives the best results may be  retained.

The test was done with the 9 images of the test file (directory testFractureOJumbo1). Poor results are obtained in the location of the boxes that detect the fractures. In 4 cases they are acceptable, in the remaining 5 the predicted box and the box with the tagged fracture only touch or overlap slightly.



When you run this test, the  image appears with a green rectangle indicating the rectangle with which the image was labeled and in blue with the predicted rectangle.

#Comments and conclusions:

The results obtained are lower than those obtained with other models with the same data set:

https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10
https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch

The associated CNN is an adaptation of the one in the article https://medium.com/aimonks/ssd-neural-network-revolutionizing-object-detection-f655d8b4b521 simplified without considering the proposed VGG16 due to the impossibility of integrating it.
Therefore, it is reduced to a CNN with only hidden layers.
Future versions of this essay will attempt to incorporate a VGG16 of the type that appears in:
https://www.kaggle.com/code/datastrophy/vgg16-pytorch-implementation


===
References and citations:

https://medium.com/aimonks/ssd-neural-network-revolutionizing-object-detection-f655d8b4b521

https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1

@misc{
                            fracture-ov5p1_dataset,
                            title = { fracture Dataset },
                            type = { Open Source Dataset },
                            author = { landy },
                            howpublished = { \url{ https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1 } },
                            url = { https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1 },
                            journal = { Roboflow Universe },
                            publisher = { Roboflow },
                            year = { 2024 },
                            month = { apr },
                            note = { visited on 2024-06-09 },
                            }



https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/model/1 

https://www.kaggle.com/code/datastrophy/vgg16-pytorch-implementation

https://medium.com/biased-algorithms/what-is-a-hidden-layer-in-a-neural-network-e966a6b57eee

https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10

https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch
